function dz =contPLRNN2_addprob(~,z,A,W,h,dt,C,Inp)
% Code from Monfared & Durstewitz (2020), Proceedings of the 37th International 
% Conference on Machine Learning
% (c) the authors
%%
%------------------------------------
%dz=contPLRNN2_(tt,z,A,W,h,dt,C,Inp)
n=size(z,1);
%-----------------------------------
d=zeros(size(z)); d(z>0)=1; D=diag(d);
%-----------------------------------
H=A+W*D;
HH=logm(H); 
Hcont2=HH./dt;
%----------------------------------
e=eig(H);
k=0;
for j=1:n
    if e(j)==1
        k=k+1;
    end
end
%----
k
%----------------------------------
I1=eye(k); I2=zeros(n-k);
I=blkdiag(I1,I2);
J1=zeros(k);  J2=eye(n-k);
J=blkdiag(J1,J2);
%-----------------------------------
h_=-[(I+HH)*((J-H)^-1)*h]./dt;
cin=-[(I+HH)*((J-H)^-1)*(C*Inp)]./dt;
dz=Hcont2*z+h_+cin;



