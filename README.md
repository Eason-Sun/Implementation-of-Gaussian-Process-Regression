# Implementation-of-Gaussian-Process-Regression

## Objective:
This project implements Gaussian Process Regression with three different kernels (Identity, Polynomial and Gaussian) for nonlinear data from scratch (without using any existing machine learning libraries e.g. sklearn).

## Dataset:
The data used in this project corresponds to samples from a 3D surface.

### Format:
There is one row per data instance and one column per attribute. The targets are real values. The training set is already divided into 10 subsets for 10-fold cross validation.

### Data Visualization:
![Capture](https://user-images.githubusercontent.com/29167705/63808707-95a68280-c8ee-11e9-9dbf-cba62fbe893f.JPG)

## Gaussian Kernel
We can show that the Gaussian kernel k(x, x') = exp(-||x - x'||^2 / 2σ^2) can be expressed as the inner product of an infinite-dimensional feature space.

Proof: 
![Capture](https://user-images.githubusercontent.com/29167705/63811122-96421780-c8f4-11e9-9df6-418c4e57706a.JPG)

## Mean Squared Error Comparision between Different Kernels:

### Identity Kernel:
The Mean Squared Error of the test set for Identity Kernel = 1.227554

### Gaussian Kernel (w.r.t. σ):
![Capture](https://user-images.githubusercontent.com/29167705/63811421-634c5380-c8f5-11e9-9287-d4da89d5b858.JPG)

### Polynomial Kernel (w.r.t. degree):
![Capture](https://user-images.githubusercontent.com/29167705/63811485-8d057a80-c8f5-11e9-80b9-c419e1f99143.JPG)

## Time Efficiency Comparision between Different Kernels:
Since kernel technique is applied in this case, the time efficiency for Gaussian Process Regression with polynomial kernel is Ο(1) with respect to the maximum degree of monomial basis functions. Same thing holds true for gaussian kernel. In this implementation, the matrix multiplication from polynomial kernel take advantage of the Numpy optimization instead of hard coding, therefore, it is a lot quicker.

![Capture](https://user-images.githubusercontent.com/29167705/63812041-24b79880-c8f7-11e9-87ac-11781253c82d.JPG)


