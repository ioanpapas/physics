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
path=r"C:\Users\johni\Desktop\physics\images\\"

#Set the number of samples
n=[5, 50, 200, 500]
n_try=4


# Define the radius of the disk and the number of samples to generate
radius = 1

# Initialize the tables for the means and standard deviations
means=[]
std=[]

fig_all=plt.figure(figsize=(15, 7))
ax_all = fig_all.add_subplot(111)
ax_all.set_title("Histograms of all number of samples")


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

    
    means.append(np.mean(I_N_values))
    std.append(np.std(I_N_values))

    # Create plot
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)

    ax.set_title(f'Histogram of value In, using N={n[n_try]} samples')
    ax.hist(I_N_values, bins=50)
    
    plt.show()

    ax_all.hist(I_N_values, bins=50, label=f'N={n[n_try]}')

    fig.savefig(path+f'askisi5_N={n[n_try]}', dpi=1000)

ax_all.legend()
ax_all.text(1.3, 1.3, f'mean_5={"{:.5f}".format(means[0])} std_5={"{:.3f}".format(std[0])}\n mean_50={"{:.5f}".format(means[1])} std_50={"{:.3f}".format(std[1])}\n mean_200={"{:.5f}".format(means[2])} std_200={"{:.3f}".format(std[2])}\n mean_500={"{:.5f}".format(means[3])} std_500={"{:.3f}".format(std[3])}\n', ha='center', va='center', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
fig_all.savefig(path+f'askisi5_hist_all', dpi=1000)

fig2 = plt.figure(figsize=(15, 7))
ax1 = fig2.add_subplot(111)

std_error=np.array(std)/np.sqrt(np.array(n))

ax1.set_title(f'')
ax1.errorbar(n, means, yerr=std_error, fmt='ro', markersize=5, capsize=3, ecolor='g', label='Means')
ax1.plot(n,np.array(means)-np.array(std_error), '--', color='black', label='Lower Bound')
ax1.plot(n,np.array(means)+np.array(std_error), '--', color='black', label='Upper Bound')
ax1.axhline(y=0, color='blue', label='Real Value')
ax1.legend()
fig2.savefig(path+f'askisi5_means_per_N')
plt.show()
print (means, std)

fig3 = plt.figure(figsize=(15,7))
ax2=fig3.add_subplot(111)

ax2.plot(n, std/np.array(means), '-o', label='Standard error per mean')
ax2.set_title ("Standard error per mean")
fig3.savefig(path+f'askisi5_std_per_mean_N')
plt.show()
print(std/np.array(means))






