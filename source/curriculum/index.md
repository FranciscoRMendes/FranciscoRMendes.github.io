---
title: Curriculum Vitae
date: 2024-06-09 09:29:40
---

**Overall Experience Summary**

Experienced Data Scientist/ ML Engineer (8+ years) with subject expertise in building and deploying ML Algorithms built on sensor data for various use cases using advanced signal processing techniques (spectral analysis, cepstral analysis etc.). I have extensive experience building classifiers and regression models on all kinds of sensor data, most of my project experience has involved the 5 steps below:

1.  **\[Engineering Context\]** Gaining an understanding of the physics, chemistry or biology of the problem

2.  **\[Sensors\]** Understanding the nature of the sensors deployed to estimate various features that are predictive of the problem, their sample rates, short comings and power consumption.

3.  **\[Modeling\]** Iteratively building signal processing features that differentiate classes (or are statistically significant in predicting a scalar value) that help solve the problem

4.  **\[Model Refinement\]** Taking client feedback from the real world (either business, unit economics, physics), translating that to the ML model and fine tuning it (capture more TPR at the expense of FPs, find profitable points on the ROC curve).

5.  **\[Model Deployment\]** Once client is happy with the solution I assist in model deployment, this includes implementing mathematical techniques to reduce model size (low rank optimization in case of linear transforms), reduce latency using numerical representations (FP32 to INT8) and studying the tradeoffs between size, accuracy and latency for a given problem

**Project Experience**

At Renesas Electronics I lead the deployment of our neural network deployment on our own micro-controllers. This involves the creation and maintenance of an optimized ML pipeline that can ingest client sensor data, extract features (including our own proprietary signal processing features), build/ optimize a model and deploy the model. The following are some interesting technical things that I work on day to day:

-   Finding FNNs, CNNs, DNNs that are small enough to fit on our smallest MCUs but complex enough to learn highly non-linear and complex patterns that differentiate classes.

-   Using the client's domain knowledge, to build features that translate the physics to the signal processing/ statistical domain. Sometimes this is a simple peak detector (turbine failure) but can be more complex such as differentiating two out of phase noisy sine waves (arc fault detection).

-   Many of our feature extraction techniques are simply linear transforms of the form Ax + b, which is the same as a fully connected neural network layer, I spend a lot of time optimizing these linear operations so that they can be done faster or so that the matrix A can use less space

    -   Either by moving these operations from FP32 to INT8

    -   Or approximating a series of matrix or tensor multiplies by a series of lower rank matrix multiplies

**Select Projects**

Here are a representative list of projects I have worked on in my over 8 year career as a data scientist. I have tried to be brief, but please do reach out for more information. 

**Grass level detection for a large lawnmower manufacturer**

Current electric lawnmowers consume a lot of power and are often curtailed by their battery storage capacity. Our client was interested in adjusting RPM of the cutting blade based on grass length using a camera/ LIDAR sensor to detect grass length, with the goal of using less power for shorter grass lengths. We used a low-quality camera in conjunction with LIDAR data to classify grass into one of many discrete grass lengths. I was responsible for the following:

-   Understanding the limits imposed by various sensor suites, since the goal was to save power, high quality sensors would solve the problem with high accuracy but would also consume more power

-   Reducing the memory limits of deploying a CNN based algorithm using both low rank approximation as well as carrying out most matrix multiplies in INT8.

**LENA Wearable device for Parentese detection**

John List and Dana Suskind hypothesized that speaking to children in parentese leads to increased cognitive development and early verbalization. In my role as a grad student with an expertise in cepstral analysis, signal processing and Neural networks I was responsible for the following tasks

-   Building a neural network that was capable of identifying parentese so that parents would receive a parentese score when using the device. In order to do this I had to build the following detectors

    -   Trigger/ wake point detection so that the device begins recording only when the parent is speaking

    -   Gender identification, since parentese characteristics differ in men and women

    -   Translating subjective definitions of parentese (sweeter sounding, baby talk sounding) into signal processing features, such as low rate of syllable generation, higher tones (relative to gender) and vowel elongation

    -   Working with the embedded product team to deploy the neural network on an embedded solution in a cost-effective manner (on Arduino in this case)

-   In addition to ML, I led the design of several experiments to prove our hypothesis (economic experiments are very conceptually like drug trials). This included the creation of treatment and control groups (no parentese) controlling for a variety of factors such as education, language, acoustic features (controlling for naturally lower toned voices), race and tonal languages (Chinese).

**Arc Fault Detection**

Arc faults are a leading driver of cost in the power industry. For this project, I oversaw the development of a new signal processing feature based on sinusoidal wave forms of sensor data, since in "normal" conditions sensors that detect arc fault are almost perfectly sinusoidal but in arc fault condition they are not. In addition to the technical requirements of this role, I was responsible for

-   Presenting results to the client in a very competitive environment (we were pitted against a competitor in a bake off)

-   Since false positives in arc fault involve a fairly high cost, I was responsible for inferring and tuning our model to optimize for the clients exact costs (this ended up differentiating us from our competitors).

**Skills**

My primary skills are mathematics, statistics and python (with C++ for embedded side). I am adept at reading through and understanding mathematical concepts and then translating them into python code (that resembles mathematics in an elegant way). My mathematical skills allow me to customize and troubleshoot the inner workings of any pre-made python package as well as code up an algorithm from scratch from a research paper.

On the python side, I am adept at the following key skills that I believe are necessary to support any ML effort

-   Creating a positive and happy environment that is conducive to learning (and making mistakes).

-   Writing clean, object oriented, modular code that can be tested

-   Implementing extensive unit testing for code maintenance

-   Implementing extensive regression testing to maintain accuracy, size and latency benchmarks on previously used data sets

-   CI/CD tasks

-   Project management tasks such as Jira, GitHub Issues, closing PRs, working with stakeholders.