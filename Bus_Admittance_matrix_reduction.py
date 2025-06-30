# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 20:28:07 2025

@author: Asus
"""

import torch
import math

# Define the z matrix with complex numbers
z = torch.tensor([
    [0.01+0.025j, 0.25+0.124j, 0.5+0.25j, float('inf'), 0.125+0.25j],
    [0.25+0.124j, 0.02+0.125j, 0.5+0.02j, 0.125+0.5j, float('inf')],
    [0.5+0.25j, 0.5+0.02j, 0.02+0.5j, float('inf'), 0.05+0.2j],
    [float('inf'), 0.125+0.5j, float('inf'), 0+0.1j, 0+0.5j],
    [0.125+0.25j, float('inf'), 0.05+0.2j, 0+0.5j, 0.01+0.2j]
], dtype=torch.cfloat)

# Calculate the admittance matrix y = 1/z
y = torch.zeros_like(z)
for a in range(5):
    for b in range(5):
        #if not torch.isinf(z[a, b]):
            y[a, b] = 1 / z[a, b]
        #else:
           # y[a, b] = 0  # handle inf as open circuit (0 admittance)

print("Admittance Matrix y:\n", y)

# Build the Y-bus matrix
Y = torch.zeros_like(y)
for a in range(5):
    for b in range(5):
        if a == b:
            s = 0
            for k in range(5):
                s += y[a, k]
            Y[a, b] = s
        else:
            Y[a, b] = -y[a, b]

print("\nY-bus Matrix Y:\n", Y)

# Perform Kron reduction
while True:
    try:
        n = int(input("\nEnter the number of reductions (n < 5): "))
    except ValueError:
        print("Please enter a valid integer.")
        continue
        

    if n < 5:
        Yred = Y.clone()
        for p in range(n):
            r = 5 - p - 1
            new_Y = torch.zeros((5 - n, 5 - n), dtype=torch.cfloat)
            for a in range(5 - n):
                for b in range(5 - n):
                    new_Y[a, b] = Yred[a, b] - (Yred[a, r] * Yred[r, b]) / Yred[r, r]
            Yred = new_Y
        print("\nReduced Admittance Matrix Yred:\n", Yred)
    else:
        print("Reduction not possible (n must be less than 5).")
        break
