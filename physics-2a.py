import numpy as np
import matplotlib.pyplot as plt
import math

# Set path for saving the figures
path=r"C:\Users\johni\Desktop\physics\images\\"

# Set the rate parameter
lam = 2

#Set the number of samples
n=[2, 20, 200, 500]
n_try=2

# Set the lower and upper bounds of the uniform distribution
a = 0
b = 4

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)

fig_exp=plt.figure(figsize=(15, 7))
ax_exp=fig_exp.add_subplot(111)

for n_try in range(4):
    x_mean=[]
    y_mean=[]
    for i in range(10000):
        # Generate 1000 random samples from the exponential distribution with rate parameter lam
        samples_exp = np.random.exponential(scale=1/lam, size=n[n_try])
        x_mean.append(np.mean(samples_exp))
        # Generate 1000 random samples from the uniform distribution with bounds a and b
        samples_uni = np.random.uniform(low=a, high=b, size=n[n_try])
        y_mean.append(np.mean(samples_uni))
    x_mean=np.array(x_mean)
    y_mean=np.array(y_mean)
    ax_exp.hist(x_mean, bins=10, label=f'N={n[n_try]}')
    ax.hist(y_mean, bins=10, label=f'N={n[n_try]}')

ax.set_title(f'Histogram of the uniform distribution mean of samples for different N')
ax_exp.set_title(f'Histogram of the exponential distribution mean of samples for different N')
ax.legend()
ax_exp.legend()

fig.savefig(path+f'askisi4_uniform', dpi=1000)
fig_exp.savefig(path+f'askisi4_exponential', dpi=1000)
plt.show()