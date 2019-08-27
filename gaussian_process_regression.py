import numpy as np
from numpy.linalg import inv, norm
import pandas as pd
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
import time

class GPR:

    def __init__(self, k='id', p=None):
        self.kernel = k
        self.parameter = p

    def kernel_id(self, x1, x2):
        x1_bar = np.c_[np.ones(x1.shape[0]), x1]
        x2_bar = np.c_[np.ones(x2.shape[0]), x2]
        return np.matmul(x1_bar, np.transpose(x2_bar))

    def kernel_gaussian(self, X1, X2, sd):

        def gaussian(x_i, x_j, sd):
            return math.exp(-((norm(x_i - x_j) ** 2)) / (2 * sd ** 2))

        kernel_matrix = np.zeros(shape=(len(X1), len(X2)))
        for i in range(len(X1)):
            for j in range(len(X2)):
                kernel_matrix[i][j] = gaussian(X1[i], X2[j], sd)
        return kernel_matrix

    def kernel_poly(self, x1, x2, degree):
        return (1 + x1 @ x2.T) ** degree

    def fit(self, train_data, train_target):
        self.X = train_data.values
        if self.kernel == 'id':
            gram_matrix = self.kernel_id(self.X, self.X)
        elif self.kernel == 'gaussian':
            gram_matrix = self.kernel_gaussian(self.X, self.X, self.parameter)
        elif self.kernel == 'poly':
            gram_matrix = self.kernel_poly(self.X, self.X, self.parameter)
        else:
            print('Kernel({}) is not defined.'.format(self.kernel))
            exit(0)
        n = gram_matrix.shape[0]
        self.a = inv(gram_matrix + np.eye(n)) @ train_target.values

    def predict(self, test_data):
        if self.kernel == 'id':
            k = self.kernel_id(test_data.values, self.X)
        elif self.kernel == 'gaussian':
            k = self.kernel_gaussian(test_data.values, self.X, self.parameter)
        else:
            k = self.kernel_poly(test_data.values, self.X, self.parameter)
        return k @ self.a


def MSE(x1, x2):
    return np.linalg.norm(x1 - x2)**2 / x1.shape[0]

def cross_validation(kernel, p, k_fold=10):
    MSEs = []
    for i in range(1,p+1):
        MSE_per_run = []
        for j in range(k_fold):
            df_validate_data = pd.read_csv('nonlinear-regression-dataset/trainInput' + str(j + 1) + '.csv', header=None)
            df_validate_target = pd.read_csv('nonlinear-regression-dataset/trainTarget' + str(j + 1) + '.csv', header=None)
            df_train_data, df_train_target = merge_train_files(k_fold, skip=j)
            # Create a linear regression classifier
            clf = GPR(k=kernel, p=i)
            clf.fit(df_train_data, df_train_target)
            pred = clf.predict(df_validate_data)
            MSE_per_run.append(MSE(df_validate_target, pred))
            # At the end of each k-fold cv, calculate the average MSE
            if j == k_fold - 1:
                avg_MSE = np.mean(np.array(MSE_per_run))
                MSEs.append(avg_MSE)
                print('Parameter = {}, MSE = {:8.6f}'.format(i, avg_MSE))
    # Find the index of the minimum MSE so we can get optimal Lambda by multiplying 0.1
    optimal_Lambda = np.argmin(np.array(MSEs)) + 1
    print('The best degree = ', optimal_Lambda)
    return optimal_Lambda, MSEs

# Merge multiple csv file into one data and one label data frames. Optionally, we can exclude certain files
def merge_train_files(num_of_files, skip=None):
    df_train_data = pd.DataFrame()
    df_train_label = pd.DataFrame()
    for k in range(num_of_files):
        if k == skip:
            continue
        data = pd.read_csv('nonlinear-regression-dataset/trainInput' + str(k + 1) + '.csv', header=None)
        df_train_data = df_train_data.append(data, ignore_index=True)
        label = pd.read_csv('nonlinear-regression-dataset/trainTarget' + str(k + 1) + '.csv', header=None)
        df_train_label = df_train_label.append(label, ignore_index=True)
    return df_train_data, df_train_label


df_train_data, df_train_target = merge_train_files(10)
df_test_data = pd.read_csv('nonlinear-regression-dataset/testInput.csv', header=None)
df_test_target = pd.read_csv('nonlinear-regression-dataset/testTarget.csv', header=None)

clf = GPR(k='id')
clf.fit(df_train_data, df_train_target)
pred = clf.predict(df_test_data)
print('\nThe MSE for the test set (Identity Kernel) = {:8.6f}'.format(MSE(df_test_target, pred)))

print("10-Fold Cross Validation for Gaussian Kernel:")
optimal_sigma, y = cross_validation('gaussian', 6)
clf = GPR(k='gaussian', p=optimal_sigma)
clf.fit(df_train_data, df_train_target)
pred = clf.predict(df_test_data)
print('\nThe MSE for the test set (Gaussian Kernel) = {:8.6f}'.format(MSE(df_test_target, pred)))

running_times = []
print('\nRun 20 times for each sigma:')
for sigma in range(1,10):
    running_time = []
    for i in range(20):
        start_time = time.time()
        clf = GPR(k='gaussian', p=sigma)
        clf.fit(df_train_data, df_train_target)
        pred = clf.predict(df_test_data)
        running_time.append(time.time() - start_time)
    mean_time = np.array(running_time).mean()
    running_times.append(mean_time)
    print('Average running time of sigma {}: {:6.4f}s'.format(sigma, mean_time))

# Plot the relationship between Sigma and MSE
x = [i+1 for i in range(6)]
plt.plot(x, y)
plt.xlabel('Sigma', fontsize=14)
plt.ylabel('10-Fold Cross Validation MSE', fontsize=14)
plt.title('Sigma vs Mean Squared Error', fontsize=18)
plt.show()

# Plot the relationship between Sigma and running time
x = [i for i in range(1,10)]
plt.plot(x, running_times)
plt.ylim(0, 1)
plt.xlabel('Sigma', fontsize=14)
plt.ylabel('Running Time (sec)', fontsize=14)
plt.title('Sigma vs Running Time', fontsize=18)
plt.show()


print("10-Fold Cross Validation for Polynomial Kernel:")
optimal_degree, y = cross_validation('poly', 4)
clf = GPR(k='poly', p=optimal_degree)
clf.fit(df_train_data, df_train_target)
pred = clf.predict(df_test_data)
print('\nThe MSE for the test set (Polynomial Kernel) = {:8.6f}'.format(MSE(df_test_target, pred)))

running_times = []
print('\nRun 100 times for each degee:')
for degree in range(1,30,2):
    running_time = []
    for i in range(100):
        start_time = time.time()
        clf = GPR(k='poly', p=degree)
        clf.fit(df_train_data, df_train_target)
        pred = clf.predict(df_test_data)
        running_time.append(time.time() - start_time)
    mean_time = np.array(running_time).mean()
    running_times.append(mean_time)
    print('Average running time of degree {}: {:6.4f}s'.format(degree, mean_time))

# Plot the relationship between Sigma and MSE
x = [i+1 for i in range(4)]
plt.plot(x, y)
plt.xlabel('degree', fontsize=14)
plt.ylabel('10-Fold Cross Validation MSE', fontsize=14)
plt.title('degree vs Mean Squared Error', fontsize=18)
plt.show()

# Plot the relationship between Sigma and running time
x = [i for i in range(1,30,2)]
plt.plot(x, running_times)
plt.ylim(0, 0.005)
plt.xlabel('degree', fontsize=14)
plt.ylabel('Running Time (sec)', fontsize=14)
plt.title('degree vs Running Time', fontsize=18)
plt.show()


rbf = RBF(length_scale=1)
gp = GaussianProcessRegressor(kernel=rbf)
gp.fit(df_train_data, df_train_target)
pred_ = gp.predict(df_test_data)
print(MSE(pred_, df_test_target))
