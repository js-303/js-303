import numpy as np
import math
from scipy.linalg import norm

def QRR(A):
    n, m = A.shape
    if m == 1:
        x = A[:, 0]
        alpha = norm(x, 2)
        sign = -1 if x[0] < 0 else 1
        v = x.copy()
        v[0] += sign * alpha
        norm_v = norm(v,2)
        if norm_v < 1e-12:
            v = np.zeros_like(v)
        else:
            v = v / norm_v
        W = v.reshape(-1, 1)
        Y = 2 * W.T  # Artificial placeholder to enforce norm 2
        R = np.array([[sign * alpha]])
        return R, W, Y

    else:
        left = math.floor(m/2)
        right = m - left

        # Left QR
        R_L, W_L, Y_L = QRR(A[0:n, 0:left])

        # Apply transformation to the right half
        A[0:n, left:m] = A[0:n, left:m] - W_L @ (Y_L @ A[0:n, left:m])

        # Right QR
        R_R, W_R, Y_R = QRR(A[left:n, left:m])

        # Construct X
        upper_zeros = np.zeros((left, left))
        lower_product = W_R @ (Y_R @ W_L[left:n, 0:left])
        X = W_L - np.vstack([upper_zeros, lower_product])

        # Construct R
        upper_right = A[0:left, left:m]
        lower_left = np.zeros((math.ceil(m / 2), left))
        R_0 = np.block([
            [R_L, upper_right],
            [lower_left, R_R]
            ])
        print("\n R shape: ", R_0.shape)

        # Construct W
        upper_zeros_W = np.zeros((left, right))
        W = np.hstack([
            X,
            np.vstack([
                upper_zeros_W,
                W_R
            ])
        ])

        # Construct Y
        lower_left_Y = np.zeros((right, left))
        Y = np.vstack([
            Y_L,
            np.hstack([
                lower_left_Y,
                Y_R
            ])
        ])

        return R_0, W, Y
    
A = np.array([[-1],
             [7],
             [4]])

R_0, W, Y = QRR(A.copy())

def get_Q(W,Y):
    B = Y.T @ W.T
    n, m = B.shape
    Q = np.identity(n) - B
    print("\n Q shape: ", Q.shape)
    return Q

def R_pad(R_0, A):
    n, m = A.shape
    R = np.vstack([R_0, np.zeros((n - m, m))])
    #bottom_zeros = np.zeros((n - R_0.shape[0], m))
    #R = np.vstack((R_0, bottom_zeros))
    print("padded R dims:", R.shape)
    return R

print("\nn: ",A.shape[0],"m: ",A.shape[1])
Q = get_Q(W,Y)
R = R_pad(R_0, A)

A_check = Q @ R
print("\n‣ Reconstruction Error: ||A - Q@R||_F =", np.linalg.norm(A - A_check, ord='fro'))
"""
print("R:",R)
print("\nW:", W)
print("\nY:", Y)
print("\nQ:", Q)

print("\nA:", A)
print("\nA check:", A_check)
"""
