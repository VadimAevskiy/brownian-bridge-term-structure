function [n,q_mat,dthat1, Y, er]= ll_k_test(bigtheto)
thet=[-0.0075 0.0007 1];
lamda=bigtheto(1);%-0.061;%-0.1;%-0.4;%-0.01;%0.09879;
 sigma=bigtheto(2);%%0.003;%0.02;%0.0001;
k=bigtheto(3);%%0.5;%0.4;


% lamda = bigtheto(1);
% sigma =  bigtheto(2);
% k=bigtheto(3);


T=155;
N=261;
B=zeros(N,T);
A=zeros(N,T);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
        %data
        load ITL.mat;
 
R008=emuweek(268:end,1);
R025 =emuweek(268:end,2);
R05 =emuweek(268:end,3);
R1 =emuweek(268:end,4);
R2 =emuweek(268:end,5);
R3 =emuweek(268:end,6);
R4=emuweek(268:end,7);
R5 =emuweek(268:end,8);
datataustar=emuweek(268:end,11);
  

capt=length(R5);

dthat= [R008 R025 R05 R1 R2 R3 R4 R5]/5200;
%dthat=dthat./100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for tau=2:T;% tau=T-t%  tau=T-t+1
            for n=2:N;% n=n-1
                if tau>=n
        B(n,tau)=1+B(n-1,tau-1)*k*(1-1/(tau-1));
       % A(n,tau)=lamda^2/2 + A(n-1,tau-1) - (lamda+B(n-1,tau-1)*sigma)^2/2;
                else
           % B(n,tau) =( tau)/2;
                      B(n,tau) =  B(tau,tau);
        
                end
                    A(n,tau)=lamda^2/2 + A(n-1,tau-1) - (lamda+B(n-1,tau-1)*sigma).^2/2;
         end
end

A1=A(2:end,2:end);
B1=B(2:end,2:end);
%r=0.027;
%r=R008;
%tau=(N-1)/5+1;
% tau2=tau1(2:capt+1);
    % imat=1:N-1;
     imat=[4,13,26,52,104,156,208,260];
     B2=B1(imat,17:end);
     A2=A1(imat,17:end);
     
     r=[];
     for j=1:1:capt
    r(j)=  (4*dthat(j,1)'  - A2(1,j))./ B2(1,j) ;
     end
    y=[];
   
    imat1= [1/12,3/12,6/12,1,2,3,4,5];
    
    
    for i=1:1:capt
    y(:,i)=(A2(:,i)+ B2(:,i)*r(i))./imat';
     
    end
         
    q_mat=imat1;%imat./52;
    
%         plot(q_mat, 5200*y);%figure(gcf);
%         hold on;
%     
        
        %%%%%  errors
        Y=y';
        
        Y=Y(87:end,:);
        dthat1=dthat(87:end,:);
        er= dthat1- Y;
        er_sq=er.^2;
        S1=sum(sum(er.^2));
        %S=S1^0.5;
        
       % figure;
       % plot(q_mat, 5200*er);
                % matrix er norm
        n1 = norm(er)^2;
        
        
        r1=r(87:end);
        er2=[];
        for t=1:1:51
                      er2(t)=  r(t+1) - r(t)*k*(1-1/(73--i));
        end
        
        n2=(norm(er2)./sigma)^2;
        S2=sum(er2.^2);
       % n=n1+n2;
       
   
%         S3=+ 2*T1*sigma;
%                n = S1 + S2+S3;
%         
        
        