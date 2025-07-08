import numpy as np
import math

B = np.random.uniform(low=0, high=5, size=(3,1))
#B = np.array([[1],[2]])
n,m = B.shape
print(B)

def QRR(A): 
    print("\nOA:", A)
    counter = 0
    n , m = A.shape
    print(n,m)
    if m == 1:
        a = np.linalg.norm(A,2)
        e_1 = np.zeros_like(A)
        e_1[0] = 1
        v = A+np.sign(A[0])*a*e_1
        u = v/np.linalg.norm(v,2)
        W = u
        Y = 2*u
        H = np.eye(n) - 2*u@u.T
        R=H@A
        R[np.abs(R)< 1e-9] = 0
        R = np.trim_zeros(R)
        print("\n2norm of W:", np.linalg.norm(W,2))
        print("\n2norm of Y:", np.linalg.norm(Y,2))
        print("2norm of A", np.linalg.norm(A,2))
        print("\nR",R)
    else:
        floor = m//2
        ceil = (m//2)+1
        R_L, W_L, Y_L = QRR(A[:,:floor])
        A[:,floor:] = A[:,floor:] - W_L @ (Y_L.T@A[:,floor:])
        R_R, W_R, Y_R = QRR(A[floor:n,floor:m])


        X_top_row = np.zeros((floor,floor))#W_L.shape[0]-1))
        X_bottom_row = (W_R @ (Y_R.T @ W_L[floor:,:floor]))
        X_diff = np.vstack([
            X_top_row,
            X_bottom_row
        ])
        X = (np.array(W_L) - X_diff)
        print("\nX:",X)


        R_L = np.array(R_L)
        R_top_row = np.hstack([R_L, np.array(A[:floor,floor:])])
        R_R = np.array(R_R)
        R_bottom_row = np.hstack([np.zeros((ceil-1,floor)), R_R])
        R = np.vstack([
            R_top_row,
            R_bottom_row
        ])
        print("\nR", R) 

        W_right_cols = np.vstack([
            np.zeros((floor,ceil-1)),
            W_R
        ])
        W = np.hstack([X,W_right_cols])
        print("\nW:",W)

        Y_bottom_row = np.hstack([
            np.zeros((ceil-1,floor)),
            Y_R.T
        ])
        Y = np.vstack([
            Y_L.T,
            Y_bottom_row
        ])
        Y = Y.T
        print("\nY:",Y)
    return R, W, Y
R, W, Y = QRR(B)
 


def construct_Q(W,Y):
    print(W.shape)
    print(Y.shape)
    Q = (np.eye(W.shape[0])-W@Y.T)
    return Q
Q = construct_Q(W,Y)

print(B)
R_pad = np.vstack([
    R,
    np.zeros((n-m,m))
])
print(Q@R_pad)
residual = Q @ R_pad - B
print("(Q @ R_pad - B) =", np.linalg.norm(residual))

#cases with odd number of rows
#existing residuals