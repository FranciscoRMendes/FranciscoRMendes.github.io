"""
Knowledge Distillation Toy Example
====================================
Hinton et al. (2015) — "Distilling the Knowledge in a Neural Network"

The core idea: instead of training a small student model on hard one-hot
labels, train it to match the *soft* probability distribution produced by a
large pre-trained teacher.  The teacher's soft outputs carry "dark knowledge":
information about inter-class similarities that hard labels throw away.

Dataset  : 3-class spiral (2-D, synthetic — no downloads needed)
Teacher  : 4 hidden layers × 128 units
Student  : 1 hidden layer × 16 units

Expected output (≈):
  Teacher accuracy        : ~95%
  Student (baseline)      : ~82%
  Student (distilled)     : ~89%

Dependencies: torch (pip install torch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import math

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)

# ── 1. Synthetic dataset: three interleaved spirals ───────────────────────────

def make_spirals(n_per_class: int = 400, noise: float = 0.12):
    """
    Three spirals, one per class.  Deliberately hard for a shallow network —
    the teacher can memorise the boundaries; the student benefits from being
    told *how close* each point is to the other classes.
    """
    X, y = [], []
    for cls in range(3):
        t = torch.linspace(0, 4 * math.pi, n_per_class)
        r = t / (4 * math.pi)
        angle = t + cls * (2 * math.pi / 3)
        x1 = r * torch.cos(angle) + noise * torch.randn(n_per_class)
        x2 = r * torch.sin(angle) + noise * torch.randn(n_per_class)
        X.append(torch.stack([x1, x2], dim=1))
        y.append(torch.full((n_per_class,), cls, dtype=torch.long))
    return torch.cat(X), torch.cat(y)


X, y = make_spirals()
perm = torch.randperm(len(X))
split = int(0.8 * len(X))
X_train, y_train = X[perm[:split]], y[perm[:split]]
X_test,  y_test  = X[perm[split:]], y[perm[split:]]

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test,  y_test)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=512)


# ── 2. Model definitions ───────────────────────────────────────────────────────

class TeacherNet(nn.Module):
    """Large model: 4 hidden layers × 128 units."""
    def __init__(self, in_dim: int = 2, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),   nn.ReLU(),
            nn.Linear(128, 128),   nn.ReLU(),
            nn.Linear(128, 128),   nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class StudentNet(nn.Module):
    """Small model: 1 hidden layer × 16 units (~50× fewer parameters)."""
    def __init__(self, in_dim: int = 2, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ── 3. Training routines ───────────────────────────────────────────────────────

def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / total


def train_standard(model: nn.Module, loader: DataLoader,
                   epochs: int = 200, lr: float = 1e-3) -> None:
    """Vanilla cross-entropy training."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 50 == 0:
            print(f"  epoch {epoch:>3d}  loss={loss.item():.4f}")


def distillation_loss(student_logits: torch.Tensor,
                      teacher_logits: torch.Tensor,
                      labels: torch.Tensor,
                      T: float = 4.0,
                      alpha: float = 0.7) -> torch.Tensor:
    """
    Weighted sum of two terms:
      - KL divergence between softened teacher and student distributions
        (scaled by T² to keep gradient magnitudes comparable across T values)
      - Standard cross-entropy with true one-hot labels

    T   (temperature) > 1 flattens the softmax so the student learns from
        the teacher's *confidence profile*, not just its top-1 prediction.
    alpha weights the soft-label term vs. the hard-label term.
    """
    # Soft targets from teacher
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    # Log-softmax from student (required by kl_div)
    log_soft_student = F.log_softmax(student_logits / T, dim=1)

    kl = F.kl_div(log_soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
    ce = F.cross_entropy(student_logits, labels)

    return alpha * kl + (1.0 - alpha) * ce


def train_distilled(student: nn.Module, teacher: nn.Module,
                    loader: DataLoader, epochs: int = 200,
                    lr: float = 1e-3, T: float = 4.0, alpha: float = 0.7) -> None:
    """Train student to mimic teacher's soft probability outputs."""
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    teacher.eval()
    for epoch in range(1, epochs + 1):
        student.train()
        for xb, yb in loader:
            with torch.no_grad():
                t_logits = teacher(xb)          # teacher does not update
            s_logits = student(xb)
            loss = distillation_loss(s_logits, t_logits, yb, T=T, alpha=alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 50 == 0:
            print(f"  epoch {epoch:>3d}  loss={loss.item():.4f}")


# ── 4. Run experiment ──────────────────────────────────────────────────────────

print("=" * 55)
print("  Knowledge Distillation — spiral classification")
print("=" * 55)
print(f"\nTeacher parameters : {param_count(TeacherNet()):,}")
print(f"Student parameters : {param_count(StudentNet()):,}\n")

# --- Teacher ---
print("── Step 1: Train teacher (hard labels) ──────────────")
teacher = TeacherNet()
train_standard(teacher, train_loader)
teacher_acc = accuracy(teacher, test_loader)
print(f"\n  Teacher test accuracy : {teacher_acc:.2%}\n")

# --- Student baseline (no distillation) ---
print("── Step 2: Train student baseline (hard labels) ─────")
student_baseline = StudentNet()
train_standard(student_baseline, train_loader)
baseline_acc = accuracy(student_baseline, test_loader)
print(f"\n  Student (baseline) test accuracy : {baseline_acc:.2%}\n")

# --- Student with knowledge distillation ---
print("── Step 3: Train student with distillation (T=4, α=0.7) ─")
student_kd = StudentNet()
train_distilled(student_kd, teacher, train_loader)
kd_acc = accuracy(student_kd, test_loader)
print(f"\n  Student (distilled) test accuracy : {kd_acc:.2%}\n")

# --- Summary ---
print("=" * 55)
print(f"  Teacher              : {teacher_acc:.2%}")
print(f"  Student (baseline)   : {baseline_acc:.2%}")
print(f"  Student (distilled)  : {kd_acc:.2%}")
gain = kd_acc - baseline_acc
print(f"  Gain from KD         : {gain:+.2%}")
print("=" * 55)
