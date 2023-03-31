import numpy as np
import matplotlib.pyplot as plt
import math


def f(x,y):
    '''
    The function we want to estimate the double integral using MonteCarlo

    param: x,y: the coordinates
    '''
    return (x**2)*y

# Set path for saving the figures
path=r"C:\Users\User\Desktop\Python\physics-efi\\"

#Set the number of samples
n=[5, 50, 200, 500]
n_try=4


# Define the radius of the disk and the number of samples to generate
radius = 1

# Initialize the tables for the means and standard deviations
means=[]
std=[]

for n_try in range(4):

    Ν=n[n_try]

    I_N_values=[]

    for i in range(10000):

        # Generate random polar coordinates
        r = np.sqrt(np.random.uniform(0, radius**2, size=Ν))
        theta = np.random.uniform(0, 2*np.pi, size=Ν)

        # Convert polar coordinates to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        W=f(x,y)
        I_N=np.pi*np.mean(W)

        I_N_values.append(I_N)

    
    means.append(np.mean(W))
    std.append(np.std(W))

    # Create plot
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)

    ax.set_title(f'Histogram of value In, using N={n[n_try]} samples')
    plt.hist(I_N_values, bins=50)
    plt.show()

    fig.savefig(path+f'askisi5_N={n[n_try]}', dpi=1000)


fig2 = plt.figure(figsize=(15, 7))
ax1 = fig2.add_subplot(111)

std_error=np.array(std)/np.sqrt(np.array(n))

ax1.set_title(f'')
ax1.plot(n, means, '-bo', color='red', label='Means')
ax1.plot(n,np.array(means)-np.array(std_error), '--', color='black', label='Lower Bound')
ax1.plot(n,np.array(means)+np.array(std_error), '--', color='black', label='Upper Bound')
ax1.axhline(y=0, color='blue', label='Real Value')
ax1.legend()
fig2.savefig(path+f'askisi5_means_per_N')
plt.show()
print (means, std)

fig3 = plt.figure(figsize=(15,7))
ax2=fig3.add_subplot(111)

ax2.plot(n, std_error/np.array(means), '-o', label='Standard error per mean')
ax2.set_title ("Standard error per mean")
plt.show()
print(std_error/np.array(means))






