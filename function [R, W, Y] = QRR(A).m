function [R, W, Y] = QRR(A)
    % Recursive Blocked QR using compact WY representation.
    % Inputs:
    %   A - Matrix (n x m)
    % Outputs:
    %   R - Block upper-triangular matrix
    %   W, Y - WY reflectors such that Q = I - W*Y'

    [n, m] = size(A);

    % Base case: single column
    if m == 1
        x = A(:, 1);
        e1 = zeros(n,1); e1(1) = 1;
        v = sign(x(1)) * norm(x) * e1 + x;
        v = v / norm(v);

        R = v' * A;
        W = v;
        Y = 2 * v;
        return
    end

    % Indices for block partitioning
    l = floor(m / 2);        % Left block width
    r = ceil(m / 2);         % Right block width

    %% Step 1: Left block QR
    [RL, WL, YL] = QRR(A(:, 1:l));

    %% Step 2: Update right block
    A(:, l+1:end) = A(:, l+1:end) - WL * (YL' * A(:, l+1:end));

    %% Step 3: Right-bottom QR
    [RR, WR, YR] = QRR(A(l+1:end, l+1:end));

    %% Step 4: Construct X
    X = WL - [zeros(l, l);
              WR * (YR' * WL(l+1:end, 1:l))];

    %% Step 5: Build W
    W = [X, [zeros(l, r); WR]];

    %% Step 6: Build Y
    Y = [YL;
         [zeros(r, l), YR]];

    %% Step 7: Assemble R
    R = [[RL, A(1:l, l+1:end)];
         [zeros(r, l), RR]];
end
