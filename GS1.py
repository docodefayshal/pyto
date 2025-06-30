# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 16:50:58 2025

@author: Asus
"""

import torch 

Y=torch.tensor([
    [1+2j,2+3j,4+5j],
    [2+4j,5+7j,9+5j],
    [3+4j,6+3j,9+8j]
    
    ],dtype=torch.complex64)
s_base=100

s2=complex(-256.6,-110.2)/s_base
s3=complex(-138.6,-45.2)/s_base

v1=torch.tensor(1.05+0j)
v2=torch.tensor(1+0j)
v3=torch.tensor(1+0j)

tol=1e-4 
for i in range (100):
    v2_prev=v2.clone()
    v3_prev=v3.clone()
    
    v2=((torch.conj(torch.tensor(s2))/v2_prev)-Y[1,0]*v1-Y[1,2]*v3)/Y[1,1]
    v3=((torch.conj(torch.tensor(s3))/v3_prev)-Y[2,0]*v1-Y[2,1]*v2_prev)/Y[2,2]
    
    if abs(v2-v2_prev)<tol and abs (v3-v3_prev):
        print ("converged")
        break
    print(f"iteration: v2={v2:.4f} v3={v3:.4f}")
    
print(f"The final v2={v2}")
print(f"The final v3={v3}")