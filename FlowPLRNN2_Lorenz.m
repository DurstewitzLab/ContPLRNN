% Code from Monfared & Durstewitz (2020), Proceedings of the 37th International 
% Conference on Machine Learning
%%
clear all
load LorenzPLRNN.mat

%--------------------------discrete-time system ----
 Z(:,1)=zeros(10,1);
 for t=2:10000
     Z(:,t)=A*Z(:,t-1)+W*max(Z(:,t-1),0)+h;%solution of discrete PLRNN
 t
 end
 X=B*Z; % project solution into 3d space
 %pause
%--------------------------continuous-time system ----
warning off     % suppress warning due to imaginary parts in logm
Z(:,1)=zeros(10,1);
T=10000;
ts=0.1;
dt=0.1;
tvec=0:dt:T*ts;
zcont2(:,1)=Z(:,1);
for t=2:length(tvec)    %numerical solution of continuous PLRNN by RK4
     k1=contPLRNN2_Lorenz(t,zcont2(:,t-1),A,W,h,ts);
    k2=contPLRNN2_Lorenz(t,zcont2(:,t-1)+dt/2*k1,A,W,h,ts);
    k3=contPLRNN2_Lorenz(t,zcont2(:,t-1)+dt/2*k2,A,W,h,ts);
    k4=contPLRNN2_Lorenz(t,zcont2(:,t-1)+dt*k3,A,W,h,ts);
    zcont2(:,t)=(zcont2(:,t-1)+dt/6*(k1+2*k2+2*k3+k4));
    t
end
zcont2n=B*zcont2; % project solution into 3d space
%-------------------------


figure(1), hold off cla
subplot(2,3,[1 2])
plot(tvec,zcont2n(1,:),'r','linewidth',2)
axis([300 500 -4 7])
ylabel('x_1');
set(gca,'FontSize',18);
text(280,6,'\bf{A}','Fontsize',34)
text(510,6,'\bf{B}','Fontsize',34)

subplot(2,3,[4 5])
plot((1:T)*ts,X(1,:),'bo','linewidth',2)
axis([300 500 -4 7])
xlabel('time'); ylabel('x_1');
set(gca,'FontSize',18);

subplot(2,3,[3 6])
plot3(X(1,:),X(2,:),X(3,:),'bo','linewidth',2);
hold on
plot3(zcont2n(1,:),zcont2n(2,:),zcont2n(3,:),'r','linewidth',2)
set(gca,'FontSize',18);
xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
xlim([-4 7]); ylim([-5 6]); zlim([-4 5])


%%
% (c) 2020 Zahra Monfared & Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University