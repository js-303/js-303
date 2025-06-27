import numpy as np

def QRR(A):
    n, m = A.shape

    if m == 1:
        # Base case: single column
        normA = np.linalg.norm(A, 2)
        sign = np.sign(A[0, 0]) if A[0, 0] != 0 else 1
        R = sign * normA

        # Construct Householder vector
        W = A.copy()
        W[0, 0] += sign * normA
        W /= np.linalg.norm(W, 2)
        Y = 2 * W

        return R, W, Y

    else:
        k = m // 2

        # Left part recursive call
        R_L, W_L, Y_L = QRR(A[:, :k])

        # Update right block using left Householder
        A_right = A[:, k:]
        A[:, k:] = A_right - W_L @ (Y_L.T @ A_right)

        # Right part recursive call
        R_R, W_R, Y_R = QRR(A[k:, k:])

        # Construct X
        top_zero = np.zeros((k, k))
        bottom = W_R @ (Y_R.T @ W_L[k:, :])
        X = W_L - np.vstack((top_zero, bottom))

        # Construct R
        top_row = np.hstack((R_L, A[:k, k:]))
        bottom_row = np.hstack((np.zeros((m - k, k)), R_R))
        R = np.vstack((top_row, bottom_row))

        # Construct W and Y
        W = np.hstack((X, np.vstack((np.zeros((k, m - k)), W_R))))
        Y = np.vstack((Y_L, np.hstack((np.zeros((m - k, k)), Y_R))))

        return R, W, Y
    
A = np.array([[-1,2,8],
             [7,12,9],
             [4,5,6],
             [2,17,-8]], dtype=float)

R, W, Y = QRR(A.copy())

def get_Q(W,Y):
    B = W @ Y
    n, m = B.shape
    I = np.identity(n)
    Q_T = I - B
    Q = Q_T.T
    print("\n Q shape: ", Q.shape)
    return Q


Q = get_Q(W,Y)

#A_check = Q @ R

print("R:",R)
print("\nW:", W)
print("\nY:", Y)
print("\nQ:", Q)

print("\nA:", A)
#print("\nA check:", A_check)
