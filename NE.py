# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 16:57:19 2025

@author: Asus
"""

import torch 
Y=torch.tensor([
    [0.1j,0.2j,0.3j,0.4j],
    [0.5j,0.6j,0.7j,0.8j],
    [0.9j,1.1j,1.2j,1.3j],
    [1.4j,1.5j,1.6j,1.7j],
 ])

print(f"Bus Matrix is{Y}:\n")

while Y.size(0)>1:
    k=int(input("Enter the number of node to delete:/n"))-1
    n=Y.size(0)
    Y_new=torch.zeros((n-1,n-1),dtype=Y.dtype)
    for i in range(n-1):
        for j in range (n-1):
            row=i if i<k else i+1
            col=j if j<k else j+1
            Y_new[i][j]=Y[row][col]-Y[row][k]*Y[k][col]/Y[k][k]
            
    Y=Y_new
    print(f"After elemination of node{k} new matrix is:\n")
    print(Y)
    

            
            
    
