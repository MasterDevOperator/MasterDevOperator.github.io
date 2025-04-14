# Understanding Neural Networks and CNNs

## Introduction

In this post, I’ll walk through what I’ve learned about **fully connected neural networks** and **convolutional neural networks (CNNs)**. 

These are foundational components of deep learning architectures. The concepts are drawn from the lecture slides and the fastai course and my interpretation of "how things work".

## Fully Connected Neural Networks (FCNs) !!

A **fully connected neural network**, also known as a **dense neural network**, connects each input node to every node in the next layer. The architecture typically consists of:

- **Input layer**: e.g., an image flattened into a 1D vector (e.g., 32×32×3 = 3072 pixels for RGB images)
- **Hidden layers**: each node applies a weight and bias, followed by an activation function
- **Output layer**: for classification, this might have 10 nodes for 10 classes

![](/images/Neural_NET.jpg "fast.ai's logo")


