%»ñÈ¡±³¾°1
%Imback=0;
%for i=1:35
%    Im0 = double(imread([int2str(i), '.jpg'],'jpg')); 
%    Imback = Imback+Im0 ;
%end
%Imback=medfilt2(rgb2gray(Imback/35));

Im0 = (imread('1.jpg'));
Im1 = (imread('2.jpg'));
Im2 = (imread('3.jpg'));
Im3 = (imread('4.jpg'));
Im4 = (imread('5.jpg'));
Im = (1/5)*Im0+(1/5)*Im1+(1/5)*Im2+(1/5)*Im3+(1/5)*Im4;
I_gray = rgb2gray(Im);
%Imback = double(I_gray);

%Imback0 = rgb2gray(imread('1.jpg'));
Imback=medfilt2(I_gray);
[MR,MC,Dim] = size(Imback);
%¿¨¶ûÂüÂË²¨³õÊ¼»¯
R=[[1,0]',[0,1]'];
H=[[1,0]',[0,1]'];
Q=0.01*eye(2);
P = 100*eye(2);
dt=1;
A=[[1,0]',[0,1]'];
kfinit=0;
x1=zeros(300,2);
x2=zeros(300,2);

% loop over all images
fig1=1;
fig2=2;
fig15=15;
fig3=3;
fig4=4;
fig5=5;
for i = 1 : 150
    Im0 = rgb2gray(imread([int2str(i), '.jpg'],'jpg')); 
    Im= medfilt2(Im0);
  if fig1 > 0
    figure(fig1)%creat figure with handle fig1
    clf%clear current figure
    imshow(Im)
  end
  Imwork = Im;

  %extract ball
  [cc1(i),cr1(i),cc2(i),cr2(i),flag]=extract(Imwork,Imback,fig1,fig2,fig3,fig15,i);
  if flag==0
    continue
  end

  if fig1 > 0
    figure(fig1)
    hold on
      plot(cc1(i),cr1(i),'g+')
      plot(cc2(i),cr2(i),'go')
    %eval(['saveas(gcf,''TRACK/trk',int2str(i-1),'.jpg'',''jpg'')']);  
  end

  
%   % Kalman update
%   if kfinit==0
%     xp = [MC/2,MR/2,0,0]'
%   else
%     xp=A*x1(i-1,:)'
%   end
%   kfinit=1;
%   PP = A*P*A' + Q
%   K = PP*H'*inv(H*PP*H'+R)
%   x1(i,:) = (xp + K*([cc1(i),cr1(i)]' - H*xp))';
% 
%   P = (eye(4)-K*H)*PP
%    if fig1 > 0
%     figure(fig1)
%     hold on
%     plot(x1(i,1),x1(i,2),'r+')
%    end
% 
%     % Kalman update
%   if kfinit==0
%     xp = [MC/2,MR/2,0,0]'
%   else
%     xp=A*x2(i-1,:)'
%   end
%   kfinit=1;
%   PP = A*P*A' + Q
%   K = PP*H'*inv(H*PP*H'+R)
%   x2(i,:) = (xp + K*([cc2(i),cr2(i)]' - H*xp))';
% 
%   P = (eye(4)-K*H)*PP
%    if fig1 > 0
%     figure(fig1)
%     hold on
%     plot(x2(i,1),x2(i,2),'ro')
%    end
    % Kalman update
  if kfinit==0
    xp = [MC/2,MR/2]'
  else
    xp=A*x1(i-1,:)'
  end
  kfinit=1;
  PP = A*P*A' + Q
  K = PP*H'*inv(H*PP*H'+R)
  x1(i,:) = (xp + K*([cc1(i),cr1(i)]' - H*xp))';

  P = (eye(2)-K*H)*PP
   if fig1 > 0
    figure(fig1)
    hold on
    plot(x1(i,1),x1(i,2),'r+')
   end

    % Kalman update
  if kfinit==0
    xp = [MC/2,MR/2]'
  else
    xp=A*x2(i-1,:)'
  end
  kfinit=1;
  PP = A*P*A' + Q
  K = PP*H'*inv(H*PP*H'+R)
  x2(i,:) = (xp + K*([cc2(i),cr2(i)]' - H*xp))';

  P = (eye(2)-K*H)*PP
   if fig1 > 0
    figure(fig1)
    hold on
    plot(x2(i,1),x2(i,2),'ro')
   end
end

% show positions
if fig4 > 0
  figure(fig4)
  hold on
  clf
  plot(cc1,'g*');
  hold on
  plot(x1(:,1),'r+');
  figure(fig5)
  plot(cc2,'g*');
  hold on
  plot(x2(:,1),'r+');
end