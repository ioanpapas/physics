import numpy as np
import matplotlib.pyplot as plt

path=r"C:\Users\johni\Desktop\physics\images\\"


def target_distribution(x, y):
    mean = [1, 3]
    cov = [[1**2, -0.3*1*2], [-0.3*1*2, 2**2]]
    return np.exp(-0.5 * (np.dot(np.dot(np.transpose([x - mean[0], y - mean[1]]), np.linalg.inv(cov)), [x - mean[0], y - mean[1]])))


def proposal_pdf(x, y, sigma):
    """Proposal probability density function.

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        sigma (float): The standard deviation of the proposal distribution.

    Returns:
        tuple: A tuple (x_new, y_new) with the proposed new coordinates.
    """
    x_new = x + sigma * np.random.normal()
    y_new = y + sigma * np.random.normal()
    return x_new, y_new

def metropolis_hastings(p, q, x0, y0, sigma, n):
    """Metropolis-Hastings algorithm for sampling from a target probability distribution.

    Args:
        p (function): The target probability density function.
        q (function): The proposal probability density function.
        x0 (float): The initial x-coordinate of the Markov chain.
        y0 (float): The initial y-coordinate of the Markov chain.
        sigma (float): The standard deviation of the proposal distribution.
        n (int): The number of iterations to run the algorithm.

    Returns:
        np.array: An array of shape (n+1, 2) with the samples from the target distribution.
    """
    x = np.zeros((n+1, 2))
    x[0, 0] = x0
    x[0, 1] = y0
    for i in range(n):
        x_new = q(x[i, 0], x[i, 1], sigma)
        alpha = min(1, p(x_new[0], x_new[1])/p(x[i, 0], x[i, 1]))
        u = np.random.uniform()
        if u < alpha:
            x[i+1, 0] = x_new[0]
            x[i+1, 1] = x_new[1]
        else:
            x[i+1, 0] = x[i, 0]
            x[i+1, 1] = x[i, 1]
    return x

# Define the target and proposal probability density functions.
p = target_distribution
q = proposal_pdf

# Set the initial values and standard deviation of the proposal distribution.
x0 = 0.5
y0 = 0.5
sigma = 2


fig2 = plt.figure(figsize=(15, 7))
ax1 = fig2.add_subplot(111)

# Run the Metropolis-Hastings algorithm.
n = 1000
samples_2 = metropolis_hastings(p, q, x0, y0, 2, n)
samples_05 =  metropolis_hastings(p, q, x0, y0, 0.5, n)
samples_01= metropolis_hastings(p, q, x0, y0, 0.1, n)
#print(np.mean(samples[:, 0]), np.mean(samples[:, 1]))
#print(np.cov(np.transpose(samples)))
# Plot the samples.

ax1.scatter(samples_2[:, 0], samples_2[:, 1], c='r', label="sigma=2")
ax1.scatter(samples_05[:, 0], samples_05[:, 1], c='b', label="sigma=0.5")
ax1.scatter(samples_01[:, 0], samples_01[:, 1], c='g', label="sigma=0.1")

ax1.legend()
    
ax1.set_title(f"Scatter plot of the samples")
fig2.savefig(path+f'askisi6_scatter_sigma_diafora', dpi=1000)
plt.show()