import matplotlib.pyplot as plt
import scipy
import numpy as np
from scipy import linalg
from scipy.stats import rv_discrete
from scipy.stats import poisson
from scipy.stats import norm

# ----------------------------------
#           Linear Algebra
# ----------------------------------

# a.
A = np.array([1, -2, 3, 4, 5, 6, 7, 1, 9]).reshape(3, 3)
print(A)

# b.
b = np.arange(1, 4)
print(b)
# c. Solve the linear system of equations A x = b

x = scipy.linalg.solve(A, b)
print(x)

# d. Check that your solution is correct by plugging it into the equation

print(np.matmul(A, x) == b)

# e. Repeat steps a-d using a random 3x3 matrix B (instead of the vector b)

B = np.random.random(9).reshape(3, 3)

sol = scipy.linalg.solve(A, B)
print(sol)

# Check if the solution is correct
print(np.dot(A, sol))
print(B)

# f. Solve the eigenvalue problem for the matrix A and print the eigenvalues and eigenvectors
sol_eigenvalue_A = np.linalg.eig(A)
eigenValue_A = sol_eigenvalue_A[0]
eigenVector_A = sol_eigenvalue_A[1]

print(eigenValue_A)
print(eigenVector_A)

# g. Calculate the inverse, determinant of A
inverse_A = linalg.inv(A)
print(inverse_A)
print(np.dot(A, linalg.inv(A)))

determinant_A = np.linalg.det(A)
print(determinant_A)

# h. Calculate the norm of A with different orders
print(linalg.norm(A))
print(linalg.norm(A, 'fro'))
print(linalg.norm(A, 'nuc'))
print(linalg.norm(A, np.inf))


# ----------------------------------
#           Statistics
# ----------------------------------


# a.    Create a discrete random variable with poissonian distribution and plot its probability mass function (PMF),
#       cummulative distribution function (CDF) and a histogram of 1000 random realizations of the variable

mu = 0.6

mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
x_axis = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))

# PMF and CDF
fig, ax = plt.subplots(1, 1)
plt.plot(x_axis, poisson.pmf(x_axis, mu), 'bo', ms="10", label="PMF Poisson")
plt.plot(x_axis, poisson.cdf(x_axis, mu), 'ro', label="CDF Poisson")
plt.legend(loc="best", frameon=False)
plt.show()

# Plot Histogram
values_poisson = poisson.rvs(mu, size=1000)
fig, ax = plt.subplots(1, 1)
plt.hist(values_poisson, bins=np.arange(0,20))
plt.legend(loc="best", frameon=False)
plt.show()

# b.    Create a continious random variable with normal distribution and plot its probability mass function (PMF),
#       cummulative distribution function (CDF) and a histogram of 1000 random realizations of the variable

mean, var, skew, kurt = norm.stats(moments='mvsk')
x_axis = np.arange(norm.ppf(0.01), norm.ppf(0.99))

# PMF and CDF
fig, ax = plt.subplots(1, 1)
plt.plot(x_axis, norm.pdf(x_axis), 'bo', ms="10", label="PMF Norm")
plt.plot(x_axis, norm.cdf(x_axis), 'ro', label="CDF Norm")
plt.legend(loc="best", frameon=False)
plt.show()

# Plot Histogram
values_norm = norm.rvs(size=1000)
fig, ax = plt.subplots(1, 1)
plt.hist(values_norm, bins=np.arange(0,20))
plt.legend(loc="best", frameon=False)
plt.show()