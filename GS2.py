import torch
Y=torch.tensor([
    [20-50j,-10+20j,-10+30j],
    [-10+20j,26-52j,-16+32j],
    [-10+30j,-16+32j,26-62j]
], dtype=torch.complex64)
print("The Ybus matrix:")
print(Y)
S_base=100.0
S2=complex(-400,-250)/S_base
P3=200/S_base
V3m=1.04
V1=torch.tensor(1.05 +0j)
V2=torch.tensor(1.0 +0j)
V3=torch.tensor(V3m +0j)

tol=1e-4
for i in range(200):
    V2_prev, V3_prev=V2.clone(), V3.clone()
    V2=((torch.conj(torch.tensor(S2, dtype=torch.complex64))/V2_prev)-(Y[1,0]*V1 +Y[1,2]*V3_prev))/Y[1,1]
    I3=Y[2,0]*V1 + Y[2,1]*V2 + Y[2,2]*V3_prev
    S3=V3_prev*torch.conj(I3)
    Q3=S3.imag.item()
    S3_new=complex(-P3,-Q3)
    V3=((torch.conj(torch.tensor(S3_new, dtype=torch.complex64))/V3_prev)-(Y[2,0]*V1 +Y[2,1]*V2))/Y[2,2]
    V3=(V3/torch.abs(V3))*V3m
    print(f"\n Iteration {i+1}: V2={V2:.4f} V3={V3:.4f} Q3={Q3:.4f}")
    if abs(V2-V2_prev)<tol and abs(V3-V3_prev)<tol :
        print("\nConverged.")
        break

print(f"\nThe Final value:")
print(f"V2={V2:.4f}")
print(f"V3={V3:.4f}")
print(f"S3={S3_new:.4f}")