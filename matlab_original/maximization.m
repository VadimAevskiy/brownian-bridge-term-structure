
clear;
% lamda=-0.1;
% sigma=0.003;
% k=1;%0.111;%;
    %bigtheto=[lamda sigma k];
    
    x0 = [23.944243266616340,-15.438802868320284];%1 [0.899347159088268 -10.102251999583524 0.979158284188580];%[-0.506782573557704, 0.005515032545303, 1];%[-0.1,0.003,1];
      options = optimset('LargeScale','off','FinDiffType' ,'central','TolX',1e-16,'TolFun',1e-16,'Display','iter','MaxFunEvals',25000,'MaxIter',12500);
     
   %[tstars, fstar]  = fminsearch(@ll_k,x0,options);
   [tstars, fstar,exitflag,output,grad,hessian] = fminunc(@ll_k,x0,options)
  %  [tstars, fstar,exitflag,output,grad,hessian] = fminunc(@ll_k_mod,x0,options);
    
 % [tstars, fstar]  = fminsearch(@ll_k_mod,x0,options);
%     lamdar= tstars(1);
% lamda =80*lamdar/(1+ lamdar^2)
 eee = 1e-3;

  epsmat = eee*eye(2);

  hessvec = zeros(2,1);

  for i = 1:2
    hessvec(i) = ll_k(tstars'+epsmat(:,i));
  end

  hessmatl = zeros(2,2);

  for i = 1:2

    for j = 1:i

      hessmatl(i,j) = (ll_k(tstars'+epsmat(:,i)+epsmat(:,j)) ...
                        -hessvec(i)-hessvec(j)+fstar)/(eee^2);

    end

  end

  hessmatu = hessmatl' - diag(diag(hessmatl));

  hessmatf = hessmatl + hessmatu;
bighx1= inv(hessmatf);
  sevec1 = sqrt(diag(bighx1));
  
  % Matlab's Hessian
     bighx = -inv(hessian);
% 
   sevec = sqrt(diag(bighx));
%    
   
   
   
   lamda = tstars (1);
     sigma =exp(tstars(2)); 
     
     tstar = [  lamda , sigma ]
       
         hessvec2 = zeros(2,1);

  for i = 1:2
    hessvec2(i) = ll_kse(tstar'+epsmat(:,i));
  end

  hessmatl2 = zeros(2,2);

  for i = 1:2

    for j = 1:i

      hessmatl2(i,j) = (ll_kse(tstar'+epsmat(:,i)+epsmat(:,j)) ...
                        -hessvec2(i)-hessvec2(j)+fstar)/(eee^2);

    end

  end

  hessmatu2 = hessmatl2' - diag(diag(hessmatl2));

  hessmatf2 = hessmatl2 + hessmatu2;
bighx2= -inv(hessmatf2);
  sevec2 = sqrt(diag(bighx2));
  
  k=1;
  bigtheto = [tstar k];
  [n,q_mat,dthat1, Y, er]= ll_k_test(bigtheto);
  
  capt1=length(Y(:,1));
    %figures 
   % hold off;
    
 % figure
 
dates=[7, 20, 33, 39];
 
 y_hat=dthat1(dates,1:8)
Y_dates = Y(dates,1:8);

for i=1:1:4;
    %figure(gcf)
    subplot(2,2,i)
plot(q_mat,5200*y_hat(i,1:8),'DisplayName','dthat1(1,1:8)','YDataSource','dthat1(1,1:8)');
hold on;
plot(q_mat,5200*Y_dates(i,1:8),'r','DisplayName','Y(1,1:8)','YDataSource','Y(1,1:8)');
%figure%(gcf)
end
yields=[];
y_t=[];
yields=5200*y_hat;
y_t=5200*Y_dates;

%   [U, V] = meshgrid(q_mat,1:1:capt1);
%      figure
%   mesh(U,V,Y);
% hold on;
%    % figure
%    mesh(U,V,dthat1);
%   %  hold off;
%     figure
%    surf(U,V,dthat1);
%    %hold on;
%    surf(U,V,Y);

% mesh(y);
% hold on;
% mesh(dthat1);