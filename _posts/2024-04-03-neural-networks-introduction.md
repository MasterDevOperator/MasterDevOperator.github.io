![](/images/Neural_NET.jpg "fast.ai's logo")

# Understanding Neural Networks and Deep Neural Networks

*Reference: [3Blue1Brown’s YouTube series on neural networks](https://www.youtube.com/3blue1brown)*

---

## Introduction

In this post, I’ll walk through what I’ve learned about **fully connected neural networks (FCNs)** and **deep neural networks (DNNs)**. These are foundational components of deep learning architectures. The concepts are drawn from the ELEC4630 lecture slides and the *fastai* course.

---

## Fully Connected Neural Networks (FCNs) 

A **fully connected neural network**, also known as a **fully connected perceptron**, connects every input node to every node in the next layer. Each connection has an associated weight that is updated during training.

![](/images/FCN_LECTURE_SLIDE.png "FCN")
*Reference: ELEC4360 Lecture slides, University of Queensland*
Breaking this down:

- **Input layer**: Takes in 3072 values (RGB image).
- **Hidden layer**: 100 neurons. Each input is connected to each hidden neuron.
- **Output layer**: 10 neurons, representing 10 possible classes (e.g., digits 0–9).

### But what is really happening here?

Let’s break down what each part of a fully connected network is doing:

- **\( W_1 \)**: Connects the input layer to the hidden layer.
- **\( W_2 \)**: Connects the hidden layer to the output layer.
- Each value in \( W_1 \) determines how much an input \( x_j \) influences a hidden neuron \( h_i \).
- Similarly, each value in \( W_2 \) controls how much a hidden neuron affects an output \( s_i \).

Together, these weights decide how signals flow through the network. But the real magic comes from what happens *between* the layers…


### Activation Functions: The “Light Switch” for Neurons

Without activation functions, a neural network is just a stack of linear transformations — no matter how deep it goes, it still behaves like a single big matrix multiplication. That’s not enough to model the real world.

To add non-linearity and **make neurons "fire"** only when something interesting happens, we use **activation functions** !!!.

One example of an activation function is **ReLU** (*Rectified Linear Unit*):

$$
\text{ReLU}(z) = \max(0, z)
$$

There are also others such as the Sigmoid, tanh, Leaky ReLU, Maxout and ELU.


###  What does ReLU *do*, really?

ReLU acts like a **light switch**:
- When the input signal \( z \) is **positive**, the neuron *lights up* and passes the signal to the next layer.
- When \( z \) is **zero or negative**, the neuron stays *off* — filtering out weak or irrelevant patterns.

This simple mechanism allows the network to:
- Focus on **important features**
- Ignore **noise**
- Build up **hierarchical understanding** (from edges → shapes → objects)


### Summary

Adding ReLU gives the network the ability to learn non-linear decision boundaries, making deep learning… well, deep. Without it, your neural net would be no more expressive than linear regression.



## Deep Neural Networks (DNNs) 
Now what happens when we stack more layers into the Neuaral Network. We end up with a Deep Neural Network (DNN).







