% Code from Monfared & Durstewitz (2020), Proceedings of the 37th International 
% Conference on Machine Learning
%%
clear all
close all
%-----------------------------------------
load HRFfine
load Example_winp_206.mat
%--------------------------------------------------------------------------
[q,T]=size(X); %--True observations are in X
Z=mu0+C*Inp(:,1);
for t=2:T
    Z(:,t)=A*Z(:,t-1)+W*max(0,Z(:,t-1))+h+C*Inp(:,t); %--Simulated discrete PLRNN states Z
end
%-------
M=size(Z,1);
Ezi=reshape(Z,1,numel(Z));
hZ=H*Ezi';  %--filter convolution
hEzi=reshape(hZ,M,T);
xpred=B*hEzi+J*rp'; %--Predicted observations
%
%--------------------------continuous-time system -------------------------
ts=0.1;
dt=0.01;
tvec=0:dt:T*ts-dt;
K=size(Inp,1);
resol=round(ts/dt);
InpCont=reshape(repmat(Inp,resol,1),K,length(tvec));
InpCont(:,end+1)=InpCont(:,end);

shift=round(ts/(2*dt));
InpCont=[InpCont(:,resol+1:end) InpCont(:,end-resol+1:end)];

zcont2(:,1)=Z(:,1);
%----------------fourth-order RK ------------------------------
for t=2:length(tvec)
    k1=dt*contPLRNN2_fMRI(t,zcont2(:,t-1),A,W,h,ts,C,InpCont(:,t));
    k2=dt*contPLRNN2_fMRI(t+dt/2,zcont2(:,t-1)+1/2*k1,A,W,h,ts,C,InpCont(:,t));
    k3=dt*contPLRNN2_fMRI(t+dt/2,zcont2(:,t-1)+1/2*k2,A,W,h,ts,C,InpCont(:,t));
    k4=dt*contPLRNN2_fMRI(t+dt,zcont2(:,t-1)+k3,A,W,h,ts,C,InpCont(:,t));
    zcont2(:,t)=zcont2(:,t-1)+1/6*(k1+2*k2+2*k3+k4);
    t
end
%----------------------------------------------------------------

Ezic=reshape(zcont2,1,numel(zcont2));
rpCont=reshape(repmat(rp',resol,1),size(rp,2),length(tvec));
rpCont=[rpCont(:,shift+1:end) rpCont(:,end-shift+1:end)];

hZc=Hcsp*Ezic';  % filter convolution
zcont2c=reshape(hZc,M,length(tvec));
xpredC=B*zcont2c+J*rpCont; %--Predicted continuous-time observations
xpredC2=B*zcont2c; %--Predicted continuous-time observations w/o regressors

TR=3;
r=10;
%-----------Plotting ------------------------------------------
figure(1), hold off cla
subplot(2,1,1), hold off cla
plot(TR*(0:1:(T-1)),xpred(r,:),'bo','linewidth',2)
hold on
plot(TR*(ts/dt)*tvec,xpredC(r,:),'r','linewidth',2)
plot(TR*(0:1:T-1),X(r,:),'g.','linewidth',2,'MarkerSize',15)
axis([0 TR*(T-1) -4 4])
xlabel('Time (s)'); ylabel('BOLD signal');
set(gca,'FontSize',18);
text(-TR*50,4,'\bf{A}','Fontsize',34)
text(-TR*50,-7,'\bf{B}','Fontsize',34)
legend({'discrete PLRNN','cont. PLRNN','data'},'Location','best','Orientation','horizontal','box','off');

subplot(2,1,2), hold off cla
plot(TR*(ts/dt)*tvec,xpredC2(r,:),'r.-','linewidth',2,'MarkerSize',5), hold on
plot(TR*(0:1:T-1),X(r,:),'go','linewidth',2)
xlim([0 TR*35])
xlabel('Time (s)'); ylabel('BOLD signal');
set(gca,'FontSize',18);



%%
% (c) 2020 Zahra Monfared & Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University