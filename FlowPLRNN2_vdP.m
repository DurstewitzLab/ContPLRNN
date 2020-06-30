% Code from Monfared & Durstewitz (2020), Proceedings of the 37th International 
% Conference on Machine Learning
%%
clear all
close all
%% 
load ReproVanDerPol.mat
%-----------------------------------------------
T=1000;
M=length(h);
Z=zeros(M,T);
%-------------- discrete-time system ------------
Z(:,1)=mu0{1};
%
for t=2:T
    Z(:,t)=A*Z(:,t-1)+W*max(Z(:,t-1),0)+h;
t
end
%-------------- continuous-time system ----------
ts=0.1;
dt=0.01;
tvec=0:dt:T*ts;
zcont2(:,1)=Z(:,1);
for t=2:length(tvec)
    k1=dt*contPLRNN2_vdP(t,zcont2(:,t-1),A,W,h,ts);
    k2=dt*contPLRNN2_vdP(t+dt/2,zcont2(:,t-1)+1/2*k1,A,W,h,ts);
    k3=dt*contPLRNN2_vdP(t+dt/2,zcont2(:,t-1)+1/2*k2,A,W,h,ts);
    k4=dt*contPLRNN2_vdP(t+dt,zcont2(:,t-1)+k3,A,W,h,ts);
    zcont2(:,t)=zcont2(:,t-1)+1/6*(k1+2*k2+2*k3+k4);
    t
end
%------------------Plotting -------------------

figure(1)
subplot(2,3,[1 2]), hold off cla
plot(0:ts:T*ts-ts,Z(1,:),'bo','linewidth',2)
hold on
plot(tvec,zcont2(1,:),'r','linewidth',2)
axis([30 60 -2 5])
ylabel('z_1');
set(gca,'FontSize',18);
text(27,5.5,'\bf{A}','Fontsize',34)
text(60.5,5.5,'\bf{B}','Fontsize',34)
legend({'discrete','cont.'},'FontSize',18,'Box','off','Location','best')
%------------------
subplot(2,3,[4 5]), hold off cla
plot(0:ts:T*ts-ts,Z(2,:),'bo','linewidth',2)
hold on
plot(tvec,zcont2(2,:),'r.','linewidth',2)
axis([37 39 -0.2 0.6])
ylabel('z_2'); xlabel('Time')
set(gca,'FontSize',18);

%% --- state space (2d subspace)
zrg=-2:0.2:5;
[z1,z2]=meshgrid(zrg,zrg);
zff=zeros(M,numel(z1));
zff(1,:)=z1(1:end);
zff(2,:)=z2(1:end);
Dz=ffcPLRNN_(A,W,h,ts,zff); % continuous flow field
%---------------------
subplot(2,3,[3 6]), hold off cla
r=sqrt(size(zff,2));
Z1=reshape(zff(1,:),r,r);
Z2=reshape(zff(2,:),r,r);
dZ1=reshape(Dz(1,:),r,r);
dZ2=reshape(Dz(2,:),r,r);
quiver(Z1,Z2,dZ1,dZ2,'Color','k','LineWidth',1.5,'ShowArrowHead','on','MaxHeadSize',10,'AutoScaleFactor',1.5)
%----------------------
t1d=round(T/2);
hold on, plot(Z(1,t1d:end),Z(2,t1d:end),'bo','linewidth',2);
t1c=round(length(tvec)/2);
plot(zcont2(1,t1c:end),zcont2(2,t1c:end),'r','linewidth',2);
%title('cont.')
axis([-1 5 -2 1])
xlabel('z_1'); ylabel('z_2');
set(gca,'FontSize',18);

%%
% (c) 2020 Zahra Monfared & Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University