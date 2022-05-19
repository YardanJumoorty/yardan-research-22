# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:35:07 2022

@author: yarda
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt



#Defining Spin quantum number S
S = 25





#----------------Defining the Hamiltonian-----------------------





#DEFINING MATRIX FOR Z-SPIN
i = S
zeigenvalues = []
zeigenvalues.append(i)
while i != -1*S:
    i = i -1
    zeigenvalues.append(i)
    
z_hat = np.diag(zeigenvalues)
z_hat = z_hat/S   #rescaling?





test_matrix = np.array([[complex(1,6),2],[-4,1]])
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

#defining the operator
O = z_hat

####################Diagonalizing Hamiltonian#######################


H = x_hat + z_hat.dot(z_hat)#defining hamiltonian


#diagonalizing H, H = PDP^-1
eigenvalues_H,eigenvectors_H =np.linalg.eigh(H) 
D = np.diag(eigenvalues_H) 
P = eigenvectors_H
P_inverse = np.linalg.inv(P) 

#defining time_evolved operator

def e_iHt(i,t):
    exponentiated_diagonal = []
    for eigenvalue in eigenvalues_H:
        exponentiated_diagonal.append(np.exp(complex(0,i)*t*eigenvalue))
    exponentiated_D= np.diag(exponentiated_diagonal)
    return np.matmul(P,np.matmul(exponentiated_D,P_inverse))

def time_ev_O(t):
    return np.matmul(e_iHt(-1,t),np.matmul(O,e_iHt(1, t)))

def commutator(a,b):
    return np.matmul(a,b) - np.matmul(b,a)

def OTOC(t):
    complex_conjugate_commutator = np.conjugate(commutator(time_ev_O(t),O))
    product = np.matmul(np.transpose(complex_conjugate_commutator),commutator(time_ev_O(t),O))
    trace = np.trace(product)
    return ((S**2)*trace)/(2*S+1)


