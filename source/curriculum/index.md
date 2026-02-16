---
title: Curriculum Vitae
date: 2024-06-09
---

## Overall Experience Summary

Senior Machine Learning Engineer and Applied Statistician with 8+ years of experience designing, analysing, and deploying production-grade ML systems across large-scale mobility data, autonomous vehicle (AV) analytics, and embedded sensor platforms.

My expertise spans rare-event detection, statistical efficiency, signal processing, and hardware-constrained deployment. I have built and shipped models operating under severe class imbalance, tight latency/memory constraints, and safety-sensitive environments. My work consistently integrates:

1. **Problem Formalization**  
   Translating ambiguous real-world objectives (business, safety, physics-driven constraints) into precise statistical learning problems.

2. **Signal & Sensor Analysis**  
   Understanding the physical and statistical properties of signals (spectral structure, stationarity, noise characteristics, sampling rates, hardware limitations).

3. **Statistical Modeling & Feature Design**  
   Designing discriminative features using spectral analysis, cepstral analysis, and domain-specific transforms; developing calibrated classifiers and regression systems.

4. **Rare-Event & Long-Tail Optimization**  
   Building importance-weighted and variance-aware modeling pipelines; tuning decision thresholds along ROC and PR curves to optimize for asymmetric cost structures.

5. **Production Deployment & Efficiency Engineering**  
   Deploying models in Python, C++, and embedded environments; applying low-rank approximations, quantization (FP32 → INT8), and architectural compression while studying tradeoffs between accuracy, latency, and memory footprint.

---

## Professional Experience

### Lyft — Senior Machine Learning Engineer / Data Scientist  
2023 – Present  

Design and deployment of large-scale, production ML systems for anomaly detection, safety analytics, and statistical efficiency in high-volume mobility data.

**Rare-Event Detection & Statistical Efficiency**
- Architected and led development of a ghost ride detection system, an inherently low-base-rate classification problem operating on highly imbalanced datasets.
- Applied importance sampling, probability calibration, and variance-aware modeling techniques to extract weak signals from large-scale positioning and behavioral data.
- Achieved ~60.41% prevention of ghost rides while maintaining operational precision constraints.
- Designed normalized risk scoring frameworks to ensure comparability across heterogeneous traffic environments.

**AV Safety & Operational Analytics**
- Worked with autonomous vehicle operational logs and telemetry to derive structured accident and safety detection features.
- Built large-scale Python and SQL pipelines to mine long-tail safety-relevant events from high-volume logs.
- Designed statistically rigorous evaluation frameworks to prioritize rare but high-impact safety signals under resource constraints.
- Contributed to production codebases in Python and C++, maintaining code health and reproducibility standards.

---

### Renesas Electronics / Reality AI — Senior AI Engineer, Hardware ML  
2020 – 2023  

Led development and deployment of neural network pipelines for automotive and industrial sensor platforms.

- Designed and deployed ML models across accelerometer, acoustic, voltage, temperature, pressure, and LIDAR signals.
- Built end-to-end optimized pipelines that ingest client sensor data, extract proprietary signal-processing features, train models, and deploy to MCUs.
- Implemented memory- and latency-constrained inference on DRP-AI accelerators and microcontrollers.
- Optimized linear transforms of the form Ax + b (equivalent to fully connected layers) through:
  - INT8 quantization of matrix multiplications
  - Low-rank matrix approximations
  - Numerical representation optimization
- Developed multi-objective reinforcement learning systems (Soft Actor-Critic) for energy optimization under physical constraints.
- Led and mentored a team of four engineers; enforced CI/CD, unit testing, regression testing, and structured code review processes.

---

### Boston Consulting Group (BCG X / BCG GAMMA) — Senior Data Scientist  
2022 – 2023  

- Designed statistically rigorous ML and optimization systems for industrial and retail clients.
- Built high-performance C++ and Python production pipelines.
- Developed cost-sensitive classification and regression models aligned to client unit economics.
- Presented technical findings to executive stakeholders in high-stakes competitive settings.

---

### University of Chicago — Research Assistant  
Kenneth C. Griffin Department of Economics  
2020 – 2022  

- Designed and deployed an embedded audio ML pipeline (<512kB) for parentese detection on LENA devices.
- Implemented wake detection, gender classification, and acoustic feature extraction using cepstral and spectral methods.
- Translated subjective behavioral definitions into measurable signal-processing features (pitch shifts, syllable rates, vowel elongation).
- Led experimental design and statistical validation of treatment-control interventions.

---

### Deloitte Consulting LLP — Senior Data Scientist  
2016 – 2020  

- Developed real-time anomaly detection and signal analytics pipelines for oil & gas, medical, and industrial clients.
- Modeled non-stationary time-series processes and optimized cost-sensitive decision thresholds.
- Delivered production-grade ML systems in collaboration with engineering and operations teams.

---

## Select Technical Projects

### Rare-Event Ghost Ride Detection (Lyft)
- Importance-sampled anomaly detection in large-scale mobility data.
- Calibrated probability outputs and cost-sensitive threshold optimization.
- Achieved ~60% rare-event prevention under strict operational constraints.

### AV Accident & Safety Scenario Extraction
- Long-tail event mining from AV operational logs.
- Structured feature extraction for accident and safety detection analytics.
- High-volume log processing in Python and SQL.

### Grass-Level Detection (Computer Vision + LIDAR)
- CNN-based classification of grass length for power optimization in electric lawnmowers.
- Memory reduction via low-rank approximations and INT8 quantization.
- Sensor tradeoff analysis balancing power consumption and predictive accuracy.

### Arc Fault Detection
- Designed sinusoidal deviation-based features for anomaly detection in electrical signals.
- Optimized model performance under asymmetric false-positive cost constraints.
- Won competitive bake-off based on cost-sensitive optimization.

---

## Skills

**Programming:** Python, C++, SQL, MATLAB, R, Julia  
**ML Frameworks:** PyTorch, JAX, TensorFlow, scikit-learn  
**Deployment:** ONNX, embedded toolchains, C++ inference pipelines  
**Statistical Methods:** Rare-event modeling, importance sampling, probability calibration, cost-sensitive classification, experimental design  
**Signal Processing:** Spectral analysis, cepstral analysis, time-frequency transforms  
**Systems Engineering:** CI/CD, unit testing, regression testing, code review, large multi-language codebases  

---

I combine mathematical rigor with production engineering discipline. I am comfortable deriving algorithms from first principles, implementing them from scratch, and deploying them in real-world systems under safety, latency, and resource constraints.
