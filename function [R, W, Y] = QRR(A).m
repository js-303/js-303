function [R, W, Y] = QRR(A)
    % QRR - Blocked QR factorization using compact WY form.
    % Inputs:
    %   A - Matrix to factor (n x m)
    % Outputs:
    %   R - Block upper-triangular matrix
    %   W, Y - WY representation of orthogonal Q such that Q = I - W*Y'
    
    [n, m] = size(A);
    k = floor(m / 2);
    
    %% Step 1: Factor left block
    [RL, WL, YL] = QRR_basic(A(:, 1:k));
    
    %% Step 2: Update right block using left reflectors
    A(:, k+1:m) = A(:, k+1:m) - WL * (YL * A(:, k+1:m));
    
    %% Step 3: Factor updated bottom-right block
    [RR, WR, YR] = QRR_basic(A(k+1:n, k+1:m));
    
    %% Step 4: Merge transformation: construct X
    X = WL - [zeros(k, k);
              WR * (YR * WL(k+1:n, 1:k))];
    
    %% Step 5: Build W
    W = [X, ...
         [zeros(k, m - k);
          WR]];
    
    %% Step 6: Build Y
    Y = [YL;
         [zeros(m - k, k), YR]];
    
    %% Step 7: Assemble R
    R = [[RL, A(1:k, k+1:m)];
         [zeros(m - k, k), RR]];
end
