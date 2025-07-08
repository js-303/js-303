import numpy as np

def QRR(A):
    """
    Recursive QR decomposition using divide and conquer approach
    Returns R, W, Y where Q = I - W @ Y.T
    """
    # Always work with a copy to prevent modifying input
    A = A.copy()  
    n, m = A.shape
    
    print(f"Processing matrix of shape {n}x{m}")
    
    # Base case: single column
    if m == 1:
        a = np.linalg.norm(A, 2)
        e_1 = np.zeros_like(A)
        e_1[0, 0] = 1
        
        # Handle the sign carefully to avoid cancellation
        sign_val = np.sign(A[0, 0]) if A[0, 0] != 0 else 1
        v = A + sign_val * a * e_1
        
        # Normalize v to get unit vector u
        norm_v = np.linalg.norm(v, 2)
        if norm_v < 1e-14:  # Avoid division by zero
            u = e_1
        else:
            u = v / norm_v
            
        W = u
        Y = 2 * u
        
        # Compute R directly with Householder
        R = A.copy()
        R[0, 0] = -sign_val * a  # First element becomes -sign(a[0])*||a||
        for i in range(1, n):
            R[i, 0] = 0.0  # Zero out below diagonal
            
        # Clean up tiny values
        R[np.abs(R) < 1e-13] = 0
        
        print(f"Base case: R shape {R.shape}, W shape {W.shape}, Y shape {Y.shape}")
        return R, W, Y
    
    # Recursive case
    # Split matrix more carefully - handle single column case separately
    split = m // 2
    
    print(f"Recursive: split = {split}, processing left part {n}x{split}")
    
    # Process left block
    R_L, W_L, Y_L = QRR(A[:, :split])
    
    # Update right block with left transformation
    A_R = A[:, split:].copy()  # Ensure we have a copy
    A_R = A_R - W_L @ (Y_L.T @ A_R)
    
    # For m=2, split=1, we're dealing with a special case
    # In this case, R_R should be the remaining single element in the upper right
    if split == 1 and m == 2:
        # Special case for 2-column matrices
        R_R = A_R[split:split+1, :]  # Just get the first row after split
        W_R = np.zeros((n-split, 1))  # No transformation needed for this element
        Y_R = np.zeros((1, 1))        # Identity transformation
        
        # Construct R matrix - upper triangular form
        R = np.zeros((n, m))
        R[:split, :split] = R_L[:split, :]  # Upper left from R_L
        R[:split, split:] = A_R[:split, :]  # Upper right from A_R
        
        # Construct W and Y as identity transformations (no further changes)
        W = W_L
        Y = Y_L
        
        print(f"Special case: R shape {R.shape}, W shape {W.shape}, Y shape {Y.shape}")
        return R, W, Y
    
    print(f"Recursive: processing bottom-right part {n-split}x{m-split}")
    
    # Process updated bottom-right block
    R_R, W_R, Y_R = QRR(A_R[split:, :])
    
    # Calculate X matrix for W
    X = W_L.copy()  # Start with W_L
    
    # Construct R matrix
    R = np.zeros((n, m))
    R[:split, :split] = R_L[:split, :]  # Upper left block
    R[:split, split:] = A_R[:split, :]  # Upper right block
    R[split:, split:] = R_R             # Lower right block
    
    # Construct W matrix
    W = np.zeros((n, W_L.shape[1] + W_R.shape[1]))
    W[:, :W_L.shape[1]] = X
    W[split:, W_L.shape[1]:] = W_R
    
    # Construct Y matrix
    Y = np.zeros((n, Y_L.shape[1] + Y_R.shape[1]))
    Y[:, :Y_L.shape[1]] = Y_L
    Y[split:, Y_L.shape[1]:] = Y_R
    
    print(f"Recursive return: R shape {R.shape}, W shape {W.shape}, Y shape {Y.shape}")
    return R, W, Y

def construct_Q(W, Y):
    """Construct Q from W and Y where Q = I - W @ Y.T"""
    return np.eye(W.shape[0]) - W @ Y.T

def test_QRR(n=3, m=2, seed=None):
    """Test QRR decomposition with a random matrix"""
    if seed is not None:
        np.random.seed(seed)
    
    # Create a test matrix
    B = np.random.uniform(low=0, high=5, size=(n, m))
    print("\nOriginal matrix B:\n", B)
    print(f"B shape: {B.shape}")
    
    # Perform QR decomposition
    R, W, Y = QRR(B)
    Q = construct_Q(W, Y)
    
    print(f"\nFinal results - Q: {Q.shape}, R: {R.shape}")
    
    # Check results
    print("\nQ matrix:\n", Q)
    print("\nR matrix:\n", R)
    print("\nQ @ R:\n", Q @ R)
    print("\nOriginal B:\n", B)
    
    # Calculate and display residual
    residual = np.linalg.norm(Q @ R - B)
    print(f"\nResidual ||Q @ R - B|| = {residual:.2e}")
    
    # Check orthogonality of Q
    ortho_error = np.linalg.norm(Q.T @ Q - np.eye(n))
    print(f"Q orthogonality error: {ortho_error:.2e}")
    
    # Compare with NumPy's QR for reference
    Q_numpy, R_numpy = np.linalg.qr(B, mode='complete')
    numpy_residual = np.linalg.norm(Q_numpy @ R_numpy - B)
    print(f"NumPy QR residual: {numpy_residual:.2e}")
    
    return residual

# Test just the case that's failing
#print("=== Test: 3x2 matrix ===")
#test_QRR(3, 2, seed=42)

#Once that works, uncomment these:
#print("\n=== Test: 3x1 matrix ===")
#test_QRR(3, 1, seed=42)

print("\n=== Test: 4x3 matrix ===")
test_QRR(120, 5, seed=42)

# print("\n=== Test: 5x5 matrix ===")
# test_QRR(5, 5, seed=42)