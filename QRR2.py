import numpy as np
import math

def QRR(A):
    n , m = A.shape
    if m == 1:
        n, m = A.shape

        # Step 1: Compute scalar R = ±‖A‖₂
        R = np.linalg.norm(A, 2)

        # Step 2: Compute a single Householder vector w from column 0
        x = A[:, 0]
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        v = x + np.sign(x[0]) * np.linalg.norm(x) * e1
        w = v / np.linalg.norm(v)
        W = w.reshape(-1, 1)  # shape (n x 1)

        # Step 3: Create matrix Y ∈ ℝ^{m × n} with each row of norm 2
        Y = np.random.randn(m, n)
        Y = (Y / np.linalg.norm(Y, axis=1, keepdims=True)) * 2
    
    else:
        print("m > 1")
        floor = m//2
        R_L, W_L, Y_L = QRR(A[:,:floor])
        print("1:",A[:,:floor])
        print("R_L:",R_L,"\nW_L:",W_L,"\nY_L:",Y_L)
        print("\n2:",A[:,floor:m])
        A[:,floor:m] = A[:,floor:m] - W_L @ (Y_L@A[:,floor:m])
        print("\n3:",A[:,:floor:m])
        R_R, W_R, Y_R = QRR(A[floor:n,floor:m])
        print("R_R:",R_R,"\nW_R:",W_R,"\nY_R:",Y_R)
        print("\n4:",A[floor:n,floor:m])
        """
        X = W_L - np.vstack([
            [np.zeros((floor,floor))],
            [W_R @ (Y_R @ W_L[floor:n,0:floor+1])]
            ])
        """
        print(np.zeros((floor,floor)).shape)
        print(np.array([W_R @ (Y_R @ W_L[floor:n,0:floor])]).shape)
        R=[]
        W=[]
        Y=[]
    return R, W, Y
    
    

B = np.random.randint(10, size=(3,2))

R, W, Y = QRR(B)

print("R (±‖A‖₂):", R)
print("W (Householder vector, 2-norm columns):\n", W)
print("Y (rows with norm = 2):\n", Y)