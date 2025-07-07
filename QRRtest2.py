import numpy as np
import math

B = np.random.uniform(low=0, high=5, size=(6,3))
#B = np.array([[1],[2]])
n,m = B.shape
print(B)

def QRR(A): 
    count = 0
    while count < 1:
        print("\nOA:", A)
        counter = 0
        n , m = A.shape
        print(n,m)
        if m == 1:
            print("m=1")
            print(A)
            print(A.T)
            print(n)
            a = np.linalg.norm(A,2)
            print("\nnorm a",a)
            e_1 = np.zeros_like(A)
            e_1[0] = 1
            v = A+np.sign(A[0])*a*e_1
            print("v: ",v)
            u = v/np.linalg.norm(v,2)
            print(u)
            W = u
            print("W",W)
            Y = 2*u
            print("Y",Y)
            H = np.eye(n) - 2*u@u.T
            R=H@A
            print(R)
            count += 1
        else:
            print("m > 1")
            print(n,m)
            floor = m//2
            ceil = (m//2)+1
            R_L, W_L, Y_L = QRR(A[:,:floor])
            threshold = 1e-9
            R_L[np.abs(R_L)< threshold] = 0
            print(R_L)
            
            print("\nA_R:",A)
            print("1:",A[:,:floor])
            print("R_L:",R_L,"\nW_L:",W_L,"\nY_L:",Y_L)
            print("\n2:",A[:,floor:m])
            A[:,floor:m] = A[:,floor:m] - W_L @ (Y_L.T@A[:,floor:m])
            print("\n3:",A[:,:floor:m])
            R_R, W_R, Y_R = QRR(A[floor:n,floor:m])
            R_R[np.abs(R_R)< threshold] = 0
            print("\nA_R:",A)
            print("R_R:",R_R,"\nW_R:",W_R,"\nY_R:",Y_R)
            print("\n4:",A[floor:n,floor:m])

            X_top_row = np.zeros((floor,floor))#W_L.shape[0]-1))
            print("\n5:",X_top_row)
            X_bottom_row = (W_R @ (Y_R.T @ W_L[floor:n,0:floor]))
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



            print("\n10",np.array(A[:floor,floor:m]))
            print("\n11",np.array(R_L))
            print(np.array(R_L).shape[1])
            if np.array(R_L).shape[1] > 1:
                R_L = np.array(R_L)
            else:
                R_L = np.array(np.trim_zeros(R_L))
            print("\n11",R_L)
            R_top_row = np.hstack([R_L, np.array(A[0:floor,floor:m])])
            print("\n8:",R_top_row)
            
            print("\n12:", np.array(R_R))
            print(np.array(R_R).shape[1])
            if np.array(R_R).shape[1] > 1:
                R_R = np.array(R_R)
            else:
                R_R = np.array(np.trim_zeros(R_R))
            #R_R = np.reshape(R_R, np.zeros((ceil-1,floor)).shape)
            print("\nR_R",R_R.size)
            print(np.zeros((ceil,floor)).size)
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
            print("Y_R:",Y_R)
            Y_bottom_row = np.hstack([
                np.zeros((ceil-1,floor)),
                Y_R.T
            ])
            print("\nY_bottom_row",Y_bottom_row)
            Y = np.vstack([
                Y_L.T,
                Y_bottom_row
            ])
            Y = Y.T
            count += 1
            print("count: ", counter)
            print("\nending m: ",m)
            print("\nending A:",A)
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