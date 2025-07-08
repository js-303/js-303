import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# QRR core functions
def QRR(A):
    """
    Recursive QR decomposition algorithm (QRR).
    """
    n, m = A.shape
    
    #Ensure n >= m
    if n < m:
        raise ValueError(f"Input matrix must have n >= m (more rows than columns), got shape {n}x{m}")
    
    #Base case: m = 1 (single column)
    if m == 1:
        #Compute conventional Householder transformation
        a = np.linalg.norm(A, 2)
        e1 = np.zeros_like(A)
        e1[0, 0] = 1
        
        #Handle the sign to avoid cancellation
        sign_val = np.sign(A[0, 0]) if A[0, 0] != 0 else 1
        v = A + sign_val * a * e1
        
        #Normalize v to get unit vector
        u = v / np.linalg.norm(v, 2)
        
        #W has unit 2-norm columns n×1
        W = u
        
        #Y has 2-norm equal to 2 rows - shape is 1×n
        Y = 2 * u.T  # Transpose to make Y 1×n
        
        #Create m×m R matrix 1x1
        R = np.array([[-sign_val * a]])
        
        #Clean R, turn very small numbers to 0
        R[np.abs(R) < 1e-13] = 0
        
        return R, W, Y
    
    #Recursive case

    #Create floor and ceiling variables
    floor_m_2 = m // 2
    ceil_m_2 = m - floor_m_2
    
    #Compute QR decomposition of left half
    left_half = A[:, :floor_m_2]
    R_L, W_L, Y_L = QRR(left_half)
    
    #Update right half of A
    A_right = A[:, floor_m_2:].copy()
    Y_L_A_right = Y_L @ A_right
    W_L_Y_L_A_right = W_L @ Y_L_A_right
    A_right = A_right - W_L_Y_L_A_right
    
    #Compute QR decomposition of bottom-right block
    A_right_bottom = A_right[floor_m_2:, :]
    R_R, W_R, Y_R = QRR(A_right_bottom)
    
    #Construct X matrix
    zeros_top = np.zeros((floor_m_2, floor_m_2))
    W_L_bottom = W_L[floor_m_2:, :]
    
    #Match Y_R and W_L_bottom dimensions
    if Y_R.shape[1] != n - floor_m_2:
        Y_R_reshaped = np.zeros((ceil_m_2, n - floor_m_2))
        min_dim = min(Y_R.shape[1], n - floor_m_2)
        Y_R_reshaped[:, :min_dim] = Y_R[:, :min_dim]
        Y_R = Y_R_reshaped
    
    Y_R_W_L_bottom = Y_R @ W_L_bottom
    X_bottom = W_R @ Y_R_W_L_bottom
    X_diff = np.vstack([zeros_top, X_bottom])
    X = W_L - X_diff
    
    #Construct R matrix m×m 
    R = np.zeros((m, m))
    R[:floor_m_2, :floor_m_2] = R_L
    R[:floor_m_2, floor_m_2:] = A_right[:floor_m_2, :]
    R[floor_m_2:, floor_m_2:] = R_R
    
    #Construct W matrix n×m
    W_right = np.vstack([np.zeros((floor_m_2, ceil_m_2)), W_R])
    W = np.hstack([X, W_right])
    
    #Construct Y matrix m×n
    Y_top = Y_L
    Y_R_padded = np.zeros((ceil_m_2, n))
    Y_R_padded[:, floor_m_2:] = Y_R
    Y = np.vstack([Y_top, Y_R_padded])
    
    #Clean R
    R[np.abs(R) < 1e-13] = 0.0
    
    return R, W, Y

def construct_Q(W, Y):
    """
    Construct Q matrix from W and Y where Q = I - WY.
    """
    n = W.shape[0]
    return np.eye(n) - W @ Y

def least_squares_qrr(A, b):
    """Solve least squares problem using QRR algorithm"""
    m, n = A.shape
    
    #Compute QR decomposition
    R, W, Y = QRR(A)
    
    #Compute Q
    Q = construct_Q(W, Y)
    
    #Solve least squares: x = R^-1 * Q^T * b
    Qb = Q.T @ b
    
    #Back-substitution to solve R*x = Q^T*b with BLAS
    """
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = Qb[i]
        for j in range(i+1, n):
            x[i] -= R[i, j] * x[j]
        x[i] /= R[i, i]
    """
    x = linalg.solve_triangular(R, Qb[:n], lower=False)
    return x

#Prep dictionaries and lists
monthly_kva = {}
yearly_kva = {}
month = []
b = []

#Insert data into into list and create A with periodic fit
with open("...", "r") as file: #insert .txt file path into ("...")
    lines = [int(line.rstrip()) for line in file]
file.close()
for i in range(len(lines)):
    monthly_kva[i+1] = lines[i]
    month.append((i+1)/12)
    b.append(lines[i])

t = np.array(month)
A = np.vstack([np.ones(len(t)),np.cos((2*np.pi)*t),np.sin((2*np.pi)*t),np.cos((4*np.pi)*t)]).T
b = np.array(b)

#QRR implementation
QR = least_squares_qrr(A, b)
print(QR)

f = lambda t: QR[0]+QR[1]*np.cos(2*np.pi*t)+QR[2]*np.sin(2*np.pi*t)+QR[3]*np.cos(4*np.pi*t)
tt = np.linspace(0, 5, 1000)
ax = plt.axes()
plt.plot(tt, f(tt), label='periodic fit')
for i in range(len(b)):
   plt.plot(t[i], b[i], "o")
plt.title("Periodic fit of monthly MWh from JAN 2005 to DEC 2009")
plt.ylabel("MWh per month")
plt.xlabel("year t")
plt.legend()
plt.show()

print(b)