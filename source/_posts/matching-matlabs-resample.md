---
title : "Matching MATLAB's resample function in Python"
date : 2024-12-17
mathjax : true
tags:
    - statistics   
    - signal processing
categories:
    - statistics
    - machine learning
---

# Matching MATLAB's resample function in Python
It is rather annoying that a fast implementation of MATLAB's resample function does not exist in Python with minimal theoretical knowledge of signal processing. This post aims to provide a simple implementation of MATLAB's resample function in Python.
With, you guessed it, zero context and therefore no theoretical knowledge of signal processing, I will attempt to implement a Python function that matches MATLAB's resample function. The function ha been tested against MATLAB's resample function using a simple example.
I might include that later. I had originally answered this on StackExchange, but it is lost because the question was deleted. 

```python
import numpy as np
from scipy.signal import resample_poly
from math import gcd
def matlab_resample(x, resample_rate, orig_sample_rate):
    """
    Resample a signal by a rational factor (p/q) to match MATLAB's `resample` function.

    Parameters:
        x (array-like): Input signal.
        p (int): Upsampling factor.
        q (int): Downsampling factor.

    Returns:
        array-like: Resampled signal.
    """
    p = resample_rate
    q = orig_sample_rate
    factor_gcd = gcd(int(p), int(q))
    p = int(p // factor_gcd)
    q = int(q // factor_gcd)

    # Ensure input is a numpy array
    x = np.asarray(x)

    # Use resample_poly to perform efficient polyphase filtering
    y = resample_poly(x, p, q, window=('kaiser', 5.0))

    # Match MATLAB's output length behavior
    output_length = int(np.ceil(len(x) * p / q))
    y = y[:output_length]

    return y
```
`