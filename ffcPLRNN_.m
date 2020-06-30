% Code from Monfared & Durstewitz (2020), Proceedings of the 37th International 
% Conference on Machine Learning
% (c) the authors
%%
function [Dz]=ffcPLRNN_(A,W,h,dt,zff)
%-----------------------------------------------------------------------------------------------
M=length(h);
Dz=zeros(size(zff));
n=size(zff,1);
%----------------------
for i=1:size(zff,2)
d=zeros(size(zff(:,i))); d(zff(:,i)>0)=1; D=diag(d);
H=A+W*D;
HH=logm(H); 
Hcont2=HH./dt;
%----------------------
e=eig(H);
k=0;
for j=1:n
    if e(j)==1
        k=k+1
    end
end
%-----------------------
I1=eye(k); I2=zeros(n-k);
I=blkdiag(I1,I2);
J1=zeros(k);  J2=eye(n-k);
J=blkdiag(J1,J2);
h_=-[(I+HH)*((J-H)^-1)*h]./dt;
Dz(:,i)=Hcont2*zff(:,i)+h_;
end