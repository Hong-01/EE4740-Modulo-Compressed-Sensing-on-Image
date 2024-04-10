function [x_rec,time] = MILP(m,n,data)
%m: measurement number
%n: length of original vector
        
% 1. project the data on A to get y
        % define measurement matrix
        variance = 1/m;
        A =randn(m, n)*sqrt(variance);
        A_norm = normalize(A,2,'norm',2);
        y = data*A_norm';
% 2. get z, z is the fraction part of y
        % %define variable
        v = fix(y);
        z = y - v;
        size_of_v = m;
        size_of_x = n;
% 3. MILP
        f = [zeros(m,1);ones(size_of_x,1); ones(size_of_x,1)];      %[v    x_p     x_n]
        
        % Constraint matrices and vectors
        Aeq = [-eye(m), A_norm, -A_norm]; 
        
        beq = z'; 
        
        % % Variable bounds
        lb = zeros(2*size_of_x + size_of_v, 1);
        ub = []; % No upper bound
        
        % Setting the index of an integer variable
        intcon = [1:m]; % Assuming v is in front of f
        
        options = optimoptions('intlinprog', 'MaxTime',30); % Set the maximum solving time to 30 seconds
        
        tic;
        % Call intlinprog to solve
        [x, fval, exitflag, output] = intlinprog(f, intcon, [], [], Aeq, beq,lb,ub,options);
        time=toc
        
        if exitflag <= 0
            disp('Solver stopped without finding an optimal solution.');
        end
        
        x_tran = x';
        x_rec = x_tran(m+1:m+n)-x_tran(m+n+1:end);  %x_rec=(x_p)-(x_n)

end