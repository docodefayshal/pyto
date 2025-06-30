# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 21:30:17 2025

@author: Asus
"""

import torch

# Ybus matrix
Y = torch.tensor([
    [20 - 50j, -10 + 20j, -10 + 30j],
    [-10 + 20j, 26 - 52j, -16 + 32j],
    [-10 + 30j, -16 + 32j, 26 - 62j]
], dtype=torch.complex64)

# Power values in pu (S = P + jQ)
S = torch.tensor([0, -256.6 - 110.2j, -138.6 - 45.2j], dtype=torch.complex64) / 100

# Initial guesses
V_mag = torch.tensor([1.05, 1.0, 1.0])
delta = torch.tensor([0.0, 0.0, 0.0])
  
for it in range(20):
    V = V_mag * torch.exp(1j * delta)
    I = Y @ V
    S_calc = V * I.conj()

    mismatch = torch.cat([
        (S[1:].real - S_calc[1:].real),
        (S[1:].imag - S_calc[1:].imag)
    ])

    print(f"\nIteration {it + 1}:")
    for i in range(3):
        print(f"V{i + 1} = {V[i].abs():.4f} ∠ {torch.rad2deg(torch.angle(V[i])).item():.2f}°")

    if torch.max(torch.abs(mismatch)) < 1e-4:
        print("\nConverged.")
        break

    # Build Jacobian
    J = torch.zeros(4, 4)
    for i in range(1, 3):
        Vi = V[i]
        Pi = S_calc[i].real.item()
        Qi = S_calc[i].imag.item()
        Yii = Y[i, i]

        for k in range(1, 3):
            Vk = V[k]
            Yik = Y[i, k]
            d = delta[i] - delta[k]

            if i == k:
                J[i - 1, k - 1] = -Qi - V_mag[i]**2 * Yii.imag
                J[i - 1, k + 1] = Pi / V_mag[i] + V_mag[i] * Yii.real
                J[i + 1, k - 1] = Pi - V_mag[i]**2 * Yii.real
                J[i + 1, k + 1] = Qi / V_mag[i] - V_mag[i] * Yii.imag
            else:
                J[i - 1, k - 1] = V_mag[i] * V_mag[k] * (
                    Yik.real * torch.sin(d) - Yik.imag * torch.cos(d))
                J[i - 1, k + 1] = V_mag[i] * (
                    Yik.real * torch.cos(d) + Yik.imag * torch.sin(d))
                J[i + 1, k - 1] = -V_mag[i] * V_mag[k] * (
                    Yik.real * torch.cos(d) + Yik.imag * torch.sin(d))
                J[i + 1, k + 1] = V_mag[i] * (
                    Yik.real * torch.sin(d) - Yik.imag * torch.cos(d))

    dx = torch.linalg.solve(J, mismatch)
    delta[1:] += dx[:2]
    V_mag[1:] += dx[2:]

# Final voltages
print("\nFinal voltages:")
V = V_mag * torch.exp(1j * delta)
for i in range(3):
    mag = V[i].abs().item()
    ang = torch.rad2deg(torch.angle(V[i])).item()
    print(f"V{i+1}: {mag:.4f} ∠ {ang:.2f}°")
