import numpy as np
import matplotlib.pyplot as plt
import math

# Set path for saving the figures
path=r"C:\Users\User\Desktop\Python\physics-efi\\"

# Set the rate parameter
lam = 2

#Set the number of samples
n=[2, 20, 200, 500, 1000]
n_try=4

# Set the lower and upper bounds of the uniform distribution
a = 0
b = 4

x_mean=[]
y_mean=[]

for i in range(10000):
    # Generate 1000 random samples from the exponential distribution with rate parameter lam
    samples_exp = np.random.exponential(scale=1/lam, size=n[n_try])
    x_mean.append(np.mean(samples_exp))
    # Generate 1000 random samples from the uniform distribution with bounds a and b
    samples_uni = np.random.uniform(low=a, high=b, size=n[n_try])
    y_mean.append(np.mean(samples_uni))
std_x=(math.sqrt((np.mean(samples_exp**2)-(np.mean(samples_exp)**2))))
std_y=(math.sqrt((np.mean(samples_uni**2)-(np.mean(samples_uni)**2))))

std_x="{:.3f}".format(std_x)
std_y="{:.3f}".format(std_y)

# Create plot
fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)

ax.set_title(f'Histogram of means for exponential and uniform distribution,\nusing N={n[n_try]} samples')
ax.hist(x_mean, bins=10, label='Exponential with lambda=2')
ax.hist(y_mean, bins=10, label='Uniform from 0 to 4')
ax.text(0.5, 0.9, f'std_x={std_x}\n std_y={std_y}', ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
ax.legend()

fig.savefig(path+f'askisi4_N={n[n_try]}', dpi=1000)

plt.show()