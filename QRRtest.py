import numpy as np
import math

B = np.random.uniform(low=-5, high=5, size=(6,1))
n,m = B.shape
print(B)
def QRR(A):
    print("\nOA:", A)
    counter = 0
    n , m = A.shape
    print(n,m)
    if m == 1:
        print("m=1")
        A = A.reshape(1, -1)
        n = A.shape[1]

        v = A.T / np.linalg.norm(A)
        v = 2 * v

        W = 0.5 * v
        Y = v.T

        R = np.linalg.norm(A, 2)

        print("2norm v:", np.linalg.norm(v))
        print("2norm W:", np.linalg.norm(W))
        print("2norm Y:", np.linalg.norm(Y, 2))
        print("R = 2norm A:", R, np.linalg.norm(A,2))
        return R,W,Y
    else:
        counter += 1
        print("m > 1")
        print(n,m)
        floor = m//2
        ceil = (m//2)+1
        R_L, W_L, Y_L = QRR(A[:,:floor])
        print("\nA_R:",A)
        print("1:",A[:,:floor])
        print("R_L:",R_L,"\nW_L:",W_L,"\nY_L:",Y_L)
        print("\n2:",A[:,floor:m])
        A[:,floor:m] = A[:,floor:m] - W_L @ (Y_L@A[:,floor:m])
        print("\n3:",A[:,:floor:m])
        R_R, W_R, Y_R = QRR(A[floor:n,floor:m])
        print("\nA_R:",A)
        print("R_R:",R_R,"\nW_R:",W_R,"\nY_R:",Y_R)
        print("\n4:",A[floor:n,floor:m])

        X_top_row = np.zeros((floor,floor))#W_L.shape[0]-1))
        print("\n5:",X_top_row)
        X_bottom_row = (W_R @ (Y_R @ W_L[floor:n,0:floor]))
        print("\n6:",X_bottom_row)
        X_diff = np.vstack([
            X_top_row,
            X_bottom_row
        ])
        print("\n7:",X_diff)
        X = (np.array(W_L) - X_diff)
        print("\nX:",X)
        print(X.shape)
        """
        X = W_L - np.vstack([
            [np.zeros((floor,floor+1))],
            [np.array(W_R @ (Y_R @ W_L[floor:n,0:floor])).T]
            ])
        """
        #print("\n5:",np.zeros((floor,floor+1)))
        #print("\n6:",np.array(W_R @ (Y_R @ W_L[floor:n,0:floor])).T)
        print(A)
        print("\n10",np.array(A[:floor,floor:m]))
        print("\n11",np.array([R_L]))
        R_L = np.reshape(np.array([R_L]), np.array(A[:floor,floor:m]).shape)
        print("\n11",R_L)
        R_top_row = np.hstack([R_L, np.array(A[0:floor,floor:m])])
        print("\n8:",R_top_row)
        print(np.zeros((ceil-1,floor)))
        R_R = np.reshape(R_R, np.zeros((ceil-1,floor)).shape)
        print(R_R)
        R_bottom_row = np.hstack([np.zeros((ceil-1,floor)), R_R])
        print("\n9:", R_bottom_row)
        R = np.vstack([
            R_top_row,
            R_bottom_row
        ])
        print("\nR", R) 
        W_right_cols = np.vstack([
            np.zeros((floor,ceil-1)),
            W_R
        ])
        print("\nW_top_row",W_right_cols)
        W = np.hstack([X,W_right_cols])
        print("\nW:",W)
        print("\nnp.zeros(ceil-1,floor)",np.zeros((ceil-1,floor)))
        Y_bottom_row = np.hstack([
            np.zeros((ceil-1,floor)),
            Y_R
        ])
        print("\nY_bottom_row",Y_bottom_row)
        Y = np.vstack([
            Y_L,
            Y_bottom_row
        ])
        print("count: ", counter)
        print("\nending m: ",m)
        print("\nending A:",A)
        return R, W, Y

R, W, Y = QRR(B)

print("R (+or- 2norm A):", R, np.linalg.norm(B,2))
print("W (2norm columns = 1):\n", W, np.linalg.norm(W,2))
print("Y (rows with 2norm = 2):\n", Y, np.linalg.norm(Y,2))

def construct_Q(W,Y):
    print(W.shape)
    print(Y.shape)
    Q = (np.eye(W.shape[0])-W@Y).T
    return Q
Q = construct_Q(W,Y)

print(B)
R_pad = np.vstack([
    R,
    np.zeros((n-m,m))
])
print(Q@R_pad)
residual = Q @ R_pad - B
print("‖Q @ R_pad - B‖_F =", np.linalg.norm(residual))

#cases with odd number of rows
#existing residuals