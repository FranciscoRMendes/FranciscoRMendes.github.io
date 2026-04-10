---
title: Curriculum Vitae
date: 2024-06-09
---

<div class="cv-header">
  <div class="cv-name-block">
    <h1>Francisco Romaldo Mendes</h1>
    <p class="cv-updated"><em>Last updated April 2026</em></p>
  </div>
  <div class="cv-contact-block">
    <a href="mailto:Mendes.franciscoromaldo@gmail.com">Mendes.franciscoromaldo@gmail.com</a><br>
    <a href="https://franciscormendes.github.io/">franciscormendes.github.io</a>
  </div>
</div>

<hr class="cv-rule">

<div class="cv-section">
<h2>Expertise</h2>

Marketplace data scientist with experience in autonomous technology and hardware. Two-sided marketplace optimisation, causal inference, experimentation, real-time perception, embedded ML, sensor fusion.
</div>

<hr class="cv-rule">

<div class="cv-section">
<h2>Experience</h2>

<div class="cv-entry">
  <div class="cv-entry-header">
    <strong>Staff ML Engineer</strong>, Lyft Business
    <span class="cv-date">2025 – Present</span>
  </div>
  <ul>
    <li>Built rider-level price elasticity framework and rewards targeting model, optimising incentive allocation under budget constraints with A/B validation (PMM, PBET).</li>
    <li>Built graph-based lookalike models (GCN, label propagation) over the Lyft marketplace graph for targeting, fraud, and incentives across product surfaces.</li>
    <li>Developed driver supply forecasting at geohash level (24h ahead) to enable forward booking for Lyft Business.</li>
    <li>Led causal inference on ad creatives using CV feature extraction (face, text, saliency) and propensity modelling to isolate true drivers of CTR.</li>
  </ul>
</div>

<div class="cv-entry">
  <div class="cv-entry-header">
    <strong>Senior AI Engineer</strong>, Renesas Electronics / Reality AI
    <span class="cv-date">2023 – 2025</span>
  </div>
  <p class="cv-subtitle"><em>Hardware Machine Learning</em></p>
  <ul>
    <li>Architected multi-sensor ML pipelines across accelerometer, acoustic, voltage, pressure, and LIDAR modalities for automotive and industrial environments.</li>
    <li>Deployed CNN and DNN models on DRP-AI accelerators, NPUs, and MCUs; compressed via INT8 quantisation and low-rank tensor factorisation.</li>
    <li>Led and mentored 4 engineers; established CI, testing, and benchmarking pipelines.</li>
  </ul>
</div>

<div class="cv-entry">
  <div class="cv-entry-header">
    <strong>Senior Data Scientist</strong>, Boston Consulting Group (BCG X / Gamma)
    <span class="cv-date">2022 – 2023</span>
  </div>
  <ul>
    <li>Built C++ production ML pipelines (perception, forecasting, recommender systems) for latency-sensitive industrial and enterprise applications.</li>
  </ul>
</div>

<div class="cv-entry">
  <div class="cv-entry-header">
    <strong>Research Assistant</strong>, University of Chicago
    <span class="cv-date">2020 – 2022</span>
  </div>
  <ul>
    <li>Built a fully embedded audio perception system under 512 kB memory: segmentation, denoising, and neural inference.</li>
  </ul>
</div>

<div class="cv-entry">
  <div class="cv-entry-header">
    <strong>Senior Data Scientist</strong>, Deloitte Consulting LLP
    <span class="cv-date">2016 – 2020</span>
  </div>
  <ul>
    <li>Built real-time C++ anomaly detection and embedded inference pipelines for industrial and energy clients.</li>
  </ul>
</div>

</div>

<hr class="cv-rule">

<div class="cv-section">
<h2>Technical Stack</h2>

<div class="cv-stack">
  <div><strong>Programming:</strong> Python, C++, MATLAB, R, Julia</div>
  <div><strong>Frameworks:</strong> PyTorch, JAX, TensorFlow</div>
  <div><strong>Deployment:</strong> ONNX, embedded toolchains, AWS, Docker</div>
  <div><strong>Domains:</strong> Computer Vision, Sensor Fusion, Marketplace ML, RL</div>
</div>
</div>

<hr class="cv-rule">

<div class="cv-section">
<h2>Education</h2>

<div class="cv-entry">
  <div class="cv-entry-header">
    <strong>University of Chicago</strong>
    <span class="cv-date">GPA: 3.8 / 4.0</span>
  </div>
  M.S. Computational Economics / Theoretical Computer Science (2022)
</div>

<div class="cv-entry">
  <div class="cv-entry-header">
    <strong>Indian Statistical Institute, Kolkata</strong>
    <span class="cv-date">Full Scholarship, Govt. of India</span>
  </div>
  M.S. Statistics (Honours), QE Specialisation (2016)
</div>

</div>

<style>
.cv-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 8px;
}
.cv-header h1 {
  font-size: 1.9rem;
  margin: 0 0 4px 0;
  border: none;
}
.cv-updated {
  margin: 0;
  font-size: 0.85rem;
  color: #888;
}
.cv-contact-block {
  text-align: right;
  font-size: 0.9rem;
  line-height: 1.7;
}
.cv-rule {
  border: none;
  border-top: 1px solid #ddd;
  margin: 18px 0;
}
.cv-section {
  margin-bottom: 8px;
}
.cv-section h2 {
  font-size: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #555;
  margin-bottom: 14px;
  border: none;
}
.cv-entry {
  margin-bottom: 18px;
}
.cv-entry-header {
  margin-bottom: 4px;
}
.cv-date {
  display: block;
  font-size: 0.85rem;
  color: #888;
  font-style: italic;
  margin-top: 1px;
}
.cv-subtitle {
  margin: 2px 0 6px 0;
  font-size: 0.9rem;
  color: #666;
}
.cv-entry ul {
  margin: 6px 0 0 0;
  padding-left: 1.4em;
}
.cv-entry ul li {
  margin-bottom: 4px;
  font-size: 0.95rem;
  line-height: 1.6;
}
.cv-stack {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px 24px;
  font-size: 0.95rem;
}
@media (max-width: 600px) {
  .cv-header { flex-direction: column; }
  .cv-contact-block { text-align: left; }
  .cv-stack { grid-template-columns: 1fr; }
}
</style>
