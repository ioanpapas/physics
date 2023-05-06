import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
path=r"C:\Users\johni\Desktop\physics\images\\"

#GIA THN ASKHSH OPOY A BAZO B KAI ANAPODA, GIATI TA EXEI ANAPODA STHN EKFONISI

#The standard variables
N=10
e=0.5

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
        y_value=np.random.normal(value, e)
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

    estA=np.append(estA, ((a+0.5)/std_beta_hat[0]))
    estB=np.append(estB, ((b-5)/std_beta_hat[1]))
    chi2_arr=np.append(chi2_arr, chi2_min)
    pvalue=np.append(pvalue, p_value)                              




