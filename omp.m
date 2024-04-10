function [x_hat,time]  = omp(m,n, data, K)
    % m: measurement number
    % n: length of original vector
    % K - Sparsity (i.e., the number of nonzero elements in the sparse solution)


% 1. project the data on A to get y
    %Define measurement matrix
    variance = 1/m;
    A =randn(m, n)*sqrt(variance);
    A_norm = normalize(A,2,'norm',2);
    y = (data*A_norm')';

% 2. OMP
    % initialize
    [m, n] = size(A); 
    x_hat = zeros(n, 1); 
    residual = y; % Initialize residuals as observation vectors
    supportSet = []; 
    
    tic;
    for k = 1:K
        % Step 1: Find the most relevant columns to the residuals
        [~, idx] = max(abs(A' * residual)); % Calculate the correlation of each column with the residuals and find the largest
        
        % Step 2: Update the support set and add a new index
        supportSet = union(supportSet, idx); 
        
        % Step 3: Least squares fit to y using columns in the support set
        A_supp = A(:, supportSet);
        x_temp = A_supp \ y; 
        
        % Step 4: Update the solution vector
        x_hat(supportSet) = x_temp;
        
        % Step 5: Update residuals
        residual = y - A * x_hat;
    end
    time=toc

    x_hat=x_hat';   %output result
end