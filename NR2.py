# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 01:54:01 2025

@author: Asus
"""

import torch

# Ybus matrix
Y = torch.tensor([
    [20-50j, -10+20j, -10+30j],
    [-10+20j, 26-52j, -16+32j],
    [-10+30j, -16+32j, 26-62j]
], dtype=torch.complex64)

# Base values
S_base = 100.0

# Given data
S2 = complex(-400, -250) / S_base    # PQ bus load
P3 = 200 / S_base                    # PV bus active power
V3_mag = 1.04                       # PV bus voltage magnitude (fixed)

# Initial guesses
V_mag = torch.tensor([1.05, 1.0, V3_mag])
delta = torch.tensor([0.0, 0.0, 0.0])

tol = 1e-4

for it in range(20):
    V = V_mag * torch.exp(1j * delta)
    I = Y @ V
    S_calc = V * I.conj()

    # Mismatches: ΔP2, ΔP3, ΔQ2 (only Q2 because Q3 unknown, V3 fixed)
    dP2 = S2.real - S_calc[1].real
    dP3 = P3 - S_calc[2].real
    dQ2 = S2.imag - S_calc[1].imag

    mismatch = torch.tensor([dP2, dP3, dQ2])

    print(f"\nIteration {it+1}:")
    for i in range(3):
        mag = V[i].abs().item()
        ang = torch.rad2deg(torch.angle(V[i])).item()
        print(f"V{i+1} = {mag:.4f} ∠ {ang:.2f}°")
    print(f"Q3 = {S_calc[2].imag:.4f}")
    print(f"Max mismatch = {torch.max(torch.abs(mismatch)):.4e}")

    if torch.max(torch.abs(mismatch)) < tol:
        print("\nConverged.")
        break

    # Jacobian matrix (3x3) — derivatives w.r.t [delta2, delta3, V2]
    J = torch.zeros((3,3))
    # Useful variables
    for i, row in enumerate([(1,1), (2,2), (1,1)]):  # for simplicity of notation
        pass
    # Manually fill Jacobian terms:
    J[0,0] = -S_calc[1].imag - V_mag[1]**2 * Y[1,1].imag
    J[0,1] = V_mag[1]*V_mag[2]*(Y[1,2].real*torch.sin(delta[1]-delta[2]) - Y[1,2].imag*torch.cos(delta[1]-delta[2]))
    J[0,2] = S_calc[1].real / V_mag[1] + V_mag[1] * Y[1,1].real
    J[1,0] = V_mag[2]*V_mag[1]*(Y[2,1].real*torch.sin(delta[2]-delta[1]) - Y[2,1].imag*torch.cos(delta[2]-delta[1]))
    J[1,1] = -S_calc[2].imag - V_mag[2]**2 * Y[2,2].imag
    # ∂P3/∂V2 (PV bus voltage fixed)
    J[1,2] = 0
    J[2,0] = S_calc[1].real - V_mag[1]**2 * Y[1,1].real
    J[2,1] = -V_mag[1]*V_mag[2]*(Y[1,2].real*torch.cos(delta[1]-delta[2]) + Y[1,2].imag*torch.sin(delta[1]-delta[2]))
    J[2,2] = S_calc[1].imag / V_mag[1] - V_mag[1] * Y[1,1].imag
    # Solve linear system
    dx = torch.linalg.solve(J, mismatch)
    # Update variables
    delta[1] += dx[0]
    delta[2] += dx[1]
    V_mag[1] += dx[2]

# Final voltages
print("\nFinal Voltages:")
V = V_mag * torch.exp(1j * delta)
for i in range(3):
    mag = V[i].abs().item()
    ang = torch.rad2deg(torch.angle(V[i])).item()
    print(f"V{i+1} = {mag:.4f} ∠ {ang:.2f}°")
