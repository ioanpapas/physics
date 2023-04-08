import numpy as np
import math
import matplotlib.pyplot as plt

# Set path for saving the figures
path=r"C:\Users\johni\Desktop\physics\images\\"

# Set parameters of Weibull distribution
k = 2
l = 1

# Generate 1000 random numbers between 0 and 1
u = np.random.rand(1000)

# Apply inverse transform function to obtain random samples from Weibull distribution
random_samples = l*(-np.log(1-u))**(1/k)

#Calculate the moments

first_moment=np.mean (random_samples)
second_moment=np.mean (random_samples**2)
third_moment=np.mean (random_samples**3)
fourth_moment=np.mean (random_samples**4)

print(first_moment, second_moment, third_moment, fourth_moment)

print (f'So the characteristic function Φ(ω)=1+jω{first_moment}+(jω^2){second_moment}/2+(jω^3){third_moment}/6+(jω^4){fourth_moment}/24')

# Plot histogram of samples

# Create plot
fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)

ax.hist(random_samples, bins=50, density=True, label='Normalized histogram of samples')

x=np.linspace(0, 3, 1000)
pdf=(k/l)*(x/l)**(k-1)*np.exp(-(x/l)**k)
ax.set_title("Histogram of samples made from Weibull distribution")
ax.plot(x,pdf, label="PDF Function")
ax.legend()
fig.savefig(path+"askisi_3")   
plt.show()
