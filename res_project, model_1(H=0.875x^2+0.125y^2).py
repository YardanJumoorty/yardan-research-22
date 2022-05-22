# -*- coding: utf-8 -*-
"""
Created on Thu May 19 21:18:56 2022

@author: yarda
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#Defining Spin quantum number S
S = 250





#----------------Defining the Hamiltonian-----------------------





#DEFINING MATRIX FOR Z-SPIN
i = S
zeigenvalues = []
zeigenvalues.append(i)
while i != -1*S:
    i = i -1
    zeigenvalues.append(i)
    
z_hat = np.diag(zeigenvalues)
z_hat = (z_hat/S)  #rescaling?






#DEFINING MATRIX FOR X-SPIN 

#Defining dirac delta function (makes life easier)
def dirac_delta(a,b):
    if a==b:
        return 1
    else:
        return 0

#defining x_hat
x_hat = np.zeros(z_hat.shape)
for a in range(1,int(2*S+2)):
    for b in range(1,int(2*S+2)):
        x_hat[a-1,b-1] = 0.5*(dirac_delta(a, b+1)+
                              dirac_delta(a+1, b))*np.sqrt((S+1)*(a+b-1)-a*b)
       
x_hat = x_hat/S   #rescaling?

#DEFINING MATRIX FOR Y-SPIN

y_hat = np.zeros(z_hat.shape)
for a in range(1,int(2*S+2)):
    for b in range(1,int(2*S+2)):
        y_hat[a-1,b-1] = 0.5*(dirac_delta(a,b+1)-
                              dirac_delta(a+1, b))*np.sqrt((S+1)*(a+b-1)-a*b)
y_hat = (y_hat/S)*complex(0,1)
       








#defining the operator
O = z_hat

####################Diagonalizing Hamiltonian#######################


H = 0.875*x_hat@(x_hat) + 0.125*y_hat@(y_hat)#defining hamiltonian

#defining dagger function (complex conjugate transpose)
def dagger(a):
    return np.conjugate(np.transpose(a))

#diagonalizing H, H = PDP^-1
eigenvalues_H,eigenvectors_H =np.linalg.eigh(H) 
D = np.diag(eigenvalues_H) 
P = eigenvectors_H
P_inverse = dagger(P)

#defining time_evolved operator

def e_iHt(i,t):
    exponentiated_diagonal= np.exp((complex(0,i)*t*eigenvalues_H*S) )
    exponentiated_D= np.diag(exponentiated_diagonal)
    return P @ exponentiated_D @ P_inverse

def time_ev_O(t):
    return e_iHt(-1,t) @ O @ e_iHt(1, t)

def commutator(a,b):
    return np.matmul(a,b) - np.matmul(b,a)

def OTOC(t):
    product = (dagger(commutator(time_ev_O(t),O))) @ commutator(time_ev_O(t),O)
    trace = np.trace(product)
    return ((S**2)*trace)/(2*S+1)



#plotting OTOC against t
t = np.linspace(0.5, 15,50)
y = []
for k in t:
    y.append(OTOC(k))


plt.semilogy(t,y, 'rx')
plt.ylabel("C(t)")
plt.xlabel("t")
plt.title('H=0.875x^2 + 0.125y^2')


#finding best fit for straight section
x_data = np.linspace(2,10,25)

y_data = []
for k in x_data:
    y_data.append(OTOC(k))


    
def model_function(t,m,c):
    return m*t+c

popt,pcov = curve_fit(model_function,x_data,np.log(y_data))

m=popt[0]
c = popt[1]

h = []
for k in x_data:
    h.append(model_function(k, m, c))
g=[]
for c in h: 
    g.append(np.e**c)
plt.plot(x_data,g)

print("gradient of line is", m)
