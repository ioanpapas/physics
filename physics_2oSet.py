import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

path=r"C:\Users\User\Desktop\Python\physics-efi\images 2nd\\"

#GIA THN ASKHSH OPOY A BAZO B KAI ANAPODA, GIATI TA EXEI ANAPODA STHN EKFONISI

#The standard variables
N=20
e=0.1

a=np.array([])
b=np.array([])
estA=np.array([])
estB=np.array([])
chi2_arr=np.array([])
pvalue=np.array([])

for i in range(10000):
    x=np.random.uniform(0, 10, size=N)
    y=np.array([])
    for value in x:
        y_value=np.random.normal(-0.5*value+5, e)
        y=np.append(y,y_value)
    

    A=np.empty((0, 2), float)
    B=y
    for value in x:
        A=np.vstack([A,[value, 1]])
    ata=np.transpose(A)@A
    atb=np.transpose(A)@B

    const=np.linalg.solve(ata, atb)

    a=np.append(a,const[0])
    b=np.append(b,const[1])

    s2 = np.sum((y - A @ const)**2)  # sum of squared residuals
    s = np.sqrt(s2 / (N-3))
    std_beta_hat = s* np.sqrt(np.diag(np.linalg.inv(A.T @ A)))

    SST = np.sum((y - np.mean(y))**2)  # total sum of squares
    df_model = 1  # degrees of freedom for the model
    df_resid = N - 2  # degrees of freedom for the residuals
    chi2_min = s2 / df_resid  # chi-square minimum
    p_value = 1 - chi2.cdf(chi2_min, df_resid)  # p-value

    estA=np.append(estA, ((const[0]+0.5)/std_beta_hat[0]))
    estB=np.append(estB, ((const[1]-5)/std_beta_hat[1]))
    chi2_arr=np.append(chi2_arr, chi2_min)
    pvalue=np.append(pvalue, p_value)

    if (i % 25==0):
        print("25 completed")                              

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
fig.suptitle("Estimation of the parameters")
ax[0].set_title(f'Distribution of parameter a')
ax[0].hist(b, bins=50)
ax[1].set_title(f'Distribution of parameter b')
ax[1].hist(a, bins=50)


fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
fig1.suptitle("Standardized residuals")
ax1[0].set_title(f'Distribution of standardized residual of parameter a')
ax1[0].hist(estB, bins=50)
ax1[1].set_title(f'Distribution of standardized residual of parameter b')
ax1[1].hist(estA, bins=50)



fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
fig2.suptitle("Statistics of estimation")
ax2[0].set_title(f'Distribution of chi squared')
ax2[0].hist(chi2_arr, bins=50)
ax2[1].set_title(f'Distribution of p-value')
ax2[1].hist(pvalue, bins=50)



fig.savefig(path+f'estimations', dpi=1000)
fig1.savefig(path+f'standardized res', dpi=1000)
fig2.savefig(path+f'statistics', dpi=1000)

plt.show()