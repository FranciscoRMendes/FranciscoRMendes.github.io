---
title: "A Manual Implementation of Quantization in PyTorch"
date: 2024-05-16
mathjax: true
tags : 
    - AI
    - Machine Learning
    - Deep Learning
    - Quantization
    - Neural Networks
    - Optimization

categories:
    - blog-post
---

# Introduction

The packaging of extremely complex techniques inside convenient wrappers
in PyTorch often makes our life very easy and removes the need to
understand the inner workings of the code. However, this obfuscates the
theory of why such things work and why they are important to us. For instance, for neither love or money, could 
I figure out what a QuantStub and a DeQuant Stub really do and how to replicate that using pen and paper. 
In embedded systems very often we have to code up certain things \"from
scratch\" as it were and sometimes PyHopper's "convenience" can be a major impediment to understanding the underlying theory.
In the code below, I will show you how to quantize a single layer of a neural network using PyTorch.
And explain each step in excruciating detail. But before that we need to understand how or why quantization is important. 

# Intuition behind Quantization

The best way to think about quantization is to think of it through an
example. Let's say you own a store and you are printing labels for the
prices of objects, but you want to economize on the number of labels you
print. Assume here for simplicity that you can print a label that shows
a price lower than the price of the product but not more. If you print
tags for 0.20 cents, you get the following table, which shows a loss of
0.97 by printing 6 labels. This obviously didn't save you much as you
might as well have printed $6$ labels with the original prices and lost
$0$ in sales.

| Price | Tags | Loss  |
|-------|------|-------|
| 1.99  | 1.8  | -0.19 |
| 2.00  | 2    | 0.00  |
| 0.59  | 0.4  | -0.19 |
| 12.30 | 12   | -0.30 |
| 8.50  | 8.4  | -0.10 |
| 8.99  | 8.8  | -0.19 |
|       | 6    | -0.97 |


Maybe we can be more aggressive, by choosing tags rounded to the nearest dollar instead,
we can obviously lose more money but we save on one whole tag!

| Price | Tags | Loss  |
|-------|------|-------|
| 1.99  | 1    | -0.99 |
| 2.00  | 2    | 0.00  |
| 0.59  | 0    | -0.59 |
| 12.30 | 12   | -0.30 |
| 8.50  | 8    | -0.50 |
| 8.99  | 8    | -0.99 |
|       | 5    | -3.37 |


How about an even more aggressive one? We round to the nearest $10$ dollars
and use just two tags. But then we are stuck with a massive loss of
$24$ dollars. 

| Price | Tags | Loss   |
|-------|------|--------|
| 1.99  | 0    | -1.99  |
| 2.00  | 0    | -2.00  |
| 0.59  | 0    | -0.59  |
| 12.30 | 10   | -2.30  |
| 8.50  | 0    | -8.50  |
| 8.99  | 0    | -8.99  |
|       | 2    | -24.37 |


In this example, the price tags represent memory units and each price
tag printed costs a certain amount of memory. Obviously, printing as
many price tags as there are goods results in no loss of money but also
the worst possible outcome as far as memory is concerned. Going the
other way reducing the number of tags results in the largest loss in
money.

# Quantization as an (Unbounded) Optimization Problem

Clearly, this calls for an optimization problem, so we can set up the
following one : let $f(x)$ be the quantization function , then the loss
is as follows,
$$L = (f(x) - x) + \lambda |\phi (X)|$$

Where $\phi(X)$ is a count of the unique values that $f(x)$ over the entire interval of
$x\in \{x_{min}, x_{max}\}$. 

### Issues with finding a solution
A popular assumption is to assume that the function is a rounding of a linear
transformation. 
The constraint that minimizes $\phi(X)$ is difficult since the function is unbounded. 
We could solve this if we knew at least two points at which we knew the expected output for the quantization problem, but 
we do not, since there is no bound on the highest tag we can print.
If we could impose a bound on the problem, we could evaluate the function 
at the two bounds and solve it. Thus setting a bound seems to solve both problem. 

# Quantization as Bounded Optimization Problem

In the previous section, our goal was to reduce the number of price tags we print, but it was not a bounded problem. 
In your average grocery story prices could run between $0$ dollars and a $1500$ dollars. Using the scheme above you could certainly print fewer labels. 
But you could also end up printing a large number of labels in absolute terms. You could do one better by pre-determining the number of labels you want to print.
Let us then, set some bounds on the number of labels we want to print, consider the labels you want to print as $x = \{-1, 0, 1, 2\}$, this is fairly aggressive. 
Again we can set up the optimization problem as follows (there is no need to minimze $\phi(X)$, the count of unique labels for now, since we are defining that ourselves),
$$L = (\text{round}(\frac{1}{s} x + z) - x)$$
where $s$ is the scale and $z$ is the zero point.
$$x_q = \text{round}(\frac{1}{s} x + z)$$
It must be true that, 
$$\text{round}(\frac{1}{s} x_{min} + z) = x_{q,min}$$
$$\text{round}(\frac{1}{s} x_{max} + z) = x_{q,max}$$
Evaluating the above equations gives us the general solution 
$$\text{round}(\frac{1}{s}*0.59 + z) = -1$$
$$\text{round}(\frac{1}{s}*12.30 + z) = 2$$
This gives us the solution,
$$s = 3.9033$$
$$z = -1$$

| Price | Label | Loss   |
|-------|-------|--------|
| 1.99  | 0     | -1.99  |
| 2     | 0     | -2     |
| 0.59  | -1    | -1.59  |
| 12.3  | 2     | -10.3  |
| 8.5   | 1     | -7.5   |
| 8.99  | 1     | -7.99  |
|       | 4     | -31.37 |



This gives the oft quoted quantization formula,
$$x_q = \text{round}(\frac{1}{s}x + z)$$
Similarly, we get reverse the formula to get the dequantization formula i.e. starting from a quantized value we can guess 
what the original value must have been, 
$$x = s(x_q -z)$$
This is obviously lossy. 

# Implication of Quantization
We have shown that given some prices, we can quantize them to a smaller set of labels. Thus saving on the cost of labels. 
What if you remembered $s$ and $z$ and then you used the dequantization formula to guess what the original price was and charge the customer
that amount? This way you can save on the number of labels, but you can get closer to the original price by just
writing down $s$ and $z$ and using the dequantization formula. We can actually do a better job with prices as well as saving on the number
of labels. However, this is lossy, and you will lose some money. In this example, we notice that we consider charging more or less than 
the actual price as a loss both ways, to keep things simple. 

| Price | Label | Loss  | DeQuant | De-q loss   |
|-------|-------|-------|---------|-------------|
| 1.99  | 0     | 1.99  | 3.903   | 1.913       |
| 2     | 0     | 2     | 3.903   | 1.903       |
| 0.59  | -1    | 1.59  | 0.000   | 0.590       |
| 12.3  | 2     | 10.3  | 11.710  | 0.590       |
| 8.5   | 1     | 7.5   | 7.807   | 0.693       |
| 8.99  | 1     | 7.99  | 7.807   | 1.183       |
|       | 4     | 31.37 |         | 6.873333333 |

# Quantization of Matrix Multiplication
Using this we can create a recipe for quantization to help us in the case of neural networks. Recall that the basic unit of a 
neural network is the operation, 
$$y = WX$$

We can apply quantization to the weights and the input. 
We can then use dequantization to get the output.

$$y = s_w(W_q-z_w)\cdot s_x(X_q-z_x)$$
$$y = s_w s_x (W-z_w) \cdot (X-z_x)$$

Here, $W_q$ and $X_q$ are quantized matrices and thus the multiplication operation
is now not between two floating point matrices $W$ and $X$ but between $W_q$ and $X_q$, 
which are two integer matrices. With this we are ready to implement quantization in PyTorch.

# Code

Consider the following original,

```python
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.fc = torch.nn.Linear(2, 2, bias=False)
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, X):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        X = self.quant(X)
        x = self.fc(X) # [[124.,  36.]
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x
```

Now consider, the manual quantization of the weights and the input. 

```python

    def prepare_model(model_fp32, input_fp32):
        # model must be set to eval mode for static quantization logic to work
        model_fp32.eval()
        model_fp32.qconfig = torch.quantization.QConfig(
            activation=MinMaxObserver.with_args(dtype=torch.quint8),
            weight=MinMaxObserver.with_args(dtype=torch.qint8)
        )
        # Prepare the model for static quantization. This inserts observers in
        # the model that will observe activation tensors during calibration.
        model_fp32_prepared = torch.quantization.prepare(model_fp32)
        model_fp32_prepared(input_fp32)

        model_int8 = torch.quantization.convert(model_fp32_prepared)
        return model_int8


def quantize_tensor_unsigned(x, scale, zero_point, num_bits=8):
    # This function mocks the PyTorch QuantStub function which quantizes the input tensor
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)
    
class QuantM2(torch.nn.Module):

    def __init__(self, model_fp32, input_fp32):
        super(QuantM2, self).__init__()
        self.fc = torch.nn.Linear(2, 2, bias=False)
        self.model_int8 = prepare_model(model_fp32, input_fp32)
        # PyTorch automatically quantizes the model for you, we will use those weights to compute a forward pass
        W_q = self.model_int8.fc.weight().int_repr().double() 
        z_w = self.model_int8.fc.weight().q_zero_point()
        self.fc.weight.data = (W_q - z_w)

    @staticmethod
    def pytorch_result(model_int8, input_fp32):
        pytorch_res = model_int8.fc(model_int8.quant(input_fp32)).int_repr().float()
        return pytorch_res

    def forward(self, x):
            input_fp32 = x
            s_x = self.model_int8.quant(input_fp32).q_scale()
            z_x = self.model_int8.quant(input_fp32).q_zero_point()
            quant_input_unsigned = quantize_tensor_unsigned(input_fp32, s_x,z_x)
            z_x = quant_input_unsigned.zero_point
            s_x = quant_input_unsigned.scale
            s_w = self.model_int8.fc.weight().q_scale()
            x1 = self.fc(quant_input_unsigned.tensor.double() - z_x)
            x1 = x1 * (s_x * s_w)
            return x1

```

Sample run code 

```python
cal_dat = torch.randn(1, 2)
model = M()
# graph mode implementation
sample_data = torch.randn(1, 2)
model(sample_data)
quant_model_2 = QuantM2(model_fp32=model, input_fp32=sample_data)
quant_model_2(sample_data)
quant_model.model_int8(sample_data) # this is the quantized model
```
