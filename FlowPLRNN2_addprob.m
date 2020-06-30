% Code from Monfared & Durstewitz (2020), Proceedings of the 37th International 
% Conference on Machine Learning
%%
clear all
close all;
%--------------------------------------
rand('state',3);


%% 
T=500;
ts=1; 
% define discrete PLRNN-----------------
A=[1 0;0 0.01];
W=[0 1;0 0];
h=[0 -0.995]';
%---------------------------------------
T=500;
C=[0 0;1 1];
Inp=zeros(2,T);
Inp(1,:)=rand(1,T);
Inp(2,100:105)=1; Inp(2,400:405)=1;
imagesc(Inp);
Z=zeros(2,T); Z(:,1)=h;
%---------------------------------------
Z(:,1)=h;
for t=2:T
    Z(:,t)=A*Z(:,t-1)+W*max(Z(:,t-1),0)+h+C*Inp(:,t);
t
end
%----------------------------
dt=0.1;
tvec=0:dt:T*ts;
InpCont=reshape(repmat(Inp,round(ts/dt),1),2,length(tvec)-1);
InpCont(:,end+1)=InpCont(:,end);
%------------------------

zcont2(:,1)=Z(:,1);
for t=2:length(tvec)
    k1=dt*contPLRNN2_addprob(t,zcont2(:,t-1),A,W,h,ts,C,InpCont(:,t));
    k2=dt*contPLRNN2_addprob(t+dt/2,zcont2(:,t-1)+1/2*k1,A,W,h,ts,C,InpCont(:,t));
    k3=dt*contPLRNN2_addprob(t+dt/2,zcont2(:,t-1)+1/2*k2,A,W,h,ts,C,InpCont(:,t));
    k4=dt*contPLRNN2_addprob(t+dt,zcont2(:,t-1)+k3,A,W,h,ts,C,InpCont(:,t));
    zcont2(:,t)=zcont2(:,t-1)+1/6*(k1+2*k2+2*k3+k4);
    t
end

%---------------Plotting ------------------------------
figure(1)
subplot(2,3,[1 2]), hold off cla
plot(tvec,zcont2(1,:),'r','linewidth',2)
hold on
plot(0:ts:T*ts-ts,Z(1,:),'bo')
ylabel('z_1');
axis([0 500 -4 8])
set(gca,'FontSize',18);
text(-50,8,'\bf{A}','Fontsize',34)
text(510,8,'\bf{B}','Fontsize',34)
%-------
subplot(2,3,[4 5]), hold off cla
plot(tvec,zcont2(2,:),'r','linewidth',2)
hold on
plot(0:ts:T*ts-ts,Z(2,:),'bo','linewidth',2)
plot(0:ts:T*ts-ts,zeros(1,ceil(T*ts)),'k--','linewidth',1.5);
xlabel('Time'); ylabel('z_2');
axis([0 500 -1.5 1.5])
set(gca,'FontSize',18);
%---------
ax=subplot(2,3,[3 6]); hold off cla
M=length(h);
zrg=-1.5:0.15:7;
[z1,z2]=meshgrid(zrg,zrg);
zff=zeros(M,numel(z1));
zff(1,:)=z1(1:end);
zff(2,:)=z2(1:end);
Dz=ffcPLRNN2_addprob(A,W,h,ts,zff); % cont. flow field
%-----------------------------------------------
r=sqrt(size(zff,2));
Z1=reshape(zff(1,:),r,r);
Z2=reshape(zff(2,:),r,r);
dZ1=reshape(Dz(1,:),r,r);
dZ2=reshape(Dz(2,:),r,r);
contourf(Z1,Z2,sqrt(dZ1.^2+dZ2.^2),'LineColor','none','LineStyle','none','LevelStep',0.05), hold on
colormap(ax,'summer');
quiver(Z1,Z2,dZ1,dZ2,'Color','k','LineWidth',1,'ShowArrowHead','on','MaxHeadSize',1)
%-----------
hold on, %plot(zrg,zeros(1,length(zrg))-0.995/(1-0.01),'y','linewidth',1);
plot(zcont2(1,:),zcont2(2,:),'r','linewidth',2);
xlabel('z_1'); ylabel('z_2');
axis([-1 7 -1.5 1.5])
set(gca,'FontSize',18);
st1=sum(Inp(1,100:105))
st2=sum(Inp(1,400:405))
st1+st2
hold on
plot(st1,-1,'go','linewidth',2,'MarkerSize',10);
plot(st1+st2,-1,'go','linewidth',2,'MarkerSize',10);



%%
% (c) 2020 Zahra Monfared & Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
