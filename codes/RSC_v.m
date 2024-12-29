function [obj, W,obj_w] = RSC_v(X, c, alpha, beta)
%% Input and output:
% [Input]
% X: the data matrix
% U: the encoding matix
% V: the orthogonal basis matrix
% W: the feature selection matrix
% [Output]
% W: the feature selection matrix
%%
[~, n] = size(X);
m = c;
ITER1 = 50;
X = NormalizeFea(X);
HY = zeros(n,c);
GY = zeros(n,m);
U=eye(n,c);
G = zeros(size(U));
V=eye(c,m);
H = zeros(n,c);
loss_c=1;
obj=[];
Mu=1e-3;
p = 1.03;

%%
for iter=1:ITER1
    [W, obj_w] = updating_w_rsc(X,U,V,H,HY,alpha,beta,Mu);
    
    %updating U
    T = X'*W*V-H*V + (HY/Mu)*V+G-(GY/Mu);
    [LT, ~, RT] = svd(T, 0);
    U = LT * RT';
    
    %updating V
    VM = W'*X*U-H'*U+((HY'*U)/Mu);
    [LV, ~, RV] = svd(VM,0);
    V =  LV * RV';
    
    %updating H
    HM = X'*W - U*V' + HY/Mu;
    for i=1:n
        norm_hm = norm(HM(i,:));
        a = ( norm_hm - 1 + sqrt((norm_hm+1)^2 - (4/Mu)))/2;
        if (Mu * norm_hm >= 1)
            H(i,:) = (1 - (1 / (1 + Mu*a + Mu*a*a))) * HM(i,:);
        else
            H(i,:) = 0;
        end
    end
    
    
    %updatingG
    G_temp = U + (GY/Mu);
    G = 0.5 * (G_temp + abs(G_temp));
    
    %updating HY
    HY = HY + Mu * (X'*W - U*V'-H);
    
    %updating GY
    GY = GY + Mu * (U - G);
    
    Mu=max(p*Mu,10^6);
    % obj
    norms = vecnorm(H, 2, 2);
    temp_1 = sum(loss_c * log(1 + (norms / loss_c)));
    temp_2 = sum(vecnorm(W*W', 1, 2)) - norm(W, 'fro')^2;
    temp_3 = X'*W-U*V'-H;
    temp_4 = U-G;
    obj_iter = temp_1 + alpha*sum(sqrt(sum(W.^2, 2))) + (beta/2)*temp_2 + trace(temp_3'*HY) + (Mu/2)* norm(temp_3, 'fro')^2 + trace(temp_4'*GY) +(Mu/2)*norm(temp_4, 'fro')^2 ;    
    
    disp(['obj:',num2str(iter),': ',num2str(obj_iter)]);   
    obj = [obj,obj_iter];
        
    if (iter>1) && (abs(obj(iter-1)-obj(iter))/abs(obj(iter-1))<1e-3)
      
        break;
    end
    
end

