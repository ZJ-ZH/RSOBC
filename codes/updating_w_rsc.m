function [W, obj_w] = updating_w_rsc(X,U,V,H,HY,alpha,beta,Mu)
%%
iter = 50;
[d, ~] = size(X);
D = eye(d);
XX = X*X';
I_d = ones(d,d);
I_1 = eye(d);
I_s = I_d-I_1;
eps = 1e-4;
obj_w = [];
%%
for iter_w = 1:iter
    
    W = pinv(alpha*D+beta*I_s+ Mu*XX)*(Mu*X*U*V' + Mu*X*H - X*HY);
    d2 = 1./ (2 * (sqrt(sum(W .* W, 2) + eps)));
    D = diag(d2);
    temp_ww = trace(ones(d)*(W*W')) - norm(W, 'fro')^2;
    
    temp_w = X'*W-U*V'-H;
    
    obj_iter_w =  alpha*sum(sqrt(sum(W.^2, 2))) + (beta/2)*temp_ww + trace(temp_w'*HY) + (Mu/2)* norm(temp_w, 'fro')^2;
    
    obj_w = [obj_w,obj_iter_w];
    
    if (iter_w>1) && (abs(obj_w(iter_w-1)-obj_w(iter_w))/abs(obj_w(iter_w-1))<1e-3)
        break;
    end
    
end
end




