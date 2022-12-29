% NPMM_code
% Original paper: Non-parametric mixture model with TV spatial regularisation and its dual expectation maximisation algorithm.
% link: http://digital-library.theiet.org/content/journals/10.1049/iet-ipr.2017.1251 .
% Doi: http://dx.doi.org/10.1049/iet-ipr.2017.1251 .
% By Shi Yan, Jun Liu,

% clear workspace
clc;clear;
close all;
addpath('auxcodes')

% mex the needed c code, depending on windows or mac
if ~ismac
    mex .\auxcodes\DensityFun.cpp
    mex .\auxcodes\DensityFun_3d_2.cpp
else
    mex ./auxcodes/DensityFun.cpp
    mex ./auxcodes/DensityFun_3d_2.cpp
end

% create saving directory
filename = datestr(now,'mmmm_dd_yyyy__HH_MM_SS');
mkdir(filename)

% load img
if ~ismac
    Ima=imread('.\images\alldone_2.png');
else
    Ima=imread('./images/alldone_2.png');
end

% set parameter
parr = 1e-2;
inner = 2000;
tau = 5e-3;
[M N H]=size(Ima);
MN=M*N;
K=2;

% preprocessing
Ima = double(Ima);
Ima = Ima - min(Ima(:));
Ima = Ima/(max(Ima(:)));

% save the preprocessed img
if ~ismac
imwrite(Ima,['.\' filename '\' 'noisy.tif'],'tiff','Resolution',300);
else
    imwrite(Ima,['./' filename '/' 'noisy.tif'],'tiff','Resolution',300);
end

% initialization of all parameters
% phi and alpha: use kmeans (the first victor of Ima)
Img = Ima(:,:,2);
for k=1:K
    sp(k,1)=1/k*max(Img(:));
end
Lab=kmeans(Img(:),K,'start',sp);
for k=1:K
    phi(:,:,k)=double(reshape(Lab,[M N])==k);
    alpha(k)=1/K;
end

% eta1, eta2 = 0;
eta1=repmat(0,[M N K]);
eta2=eta1;

% running the algorithm
for i=1:10

    reg = parr*mydiv(eta1,eta2);
    phiood = phi;
    % show current probility map
    figure(19),imshow(phi(:,:,1),[]),title(i)
    title(['Current probility map'])
    % update p(p_k^NP) according to equ (17)
    [phi,~,p]=NonPara3dEStep_smooth(Ima,phi,reg,alpha);

    
    % reinitialization, if needed
    %if mod(i,1)==0
    %eta1=repmat(0,[M N K]);
    %eta2=eta1;
    %end
    
    % Dual step, update according to equ(29)
    % phi = \alpha^k p_k^NP e^{-gamma div}
    for j=1:inner 
        
        % update eta
        [tempx,tempy] = mygradient(phi);
        eta1 = eta1 - tau*tempx;
        eta2 = eta2 - tau*tempy;
        [eta1,eta2] = Proj(eta1,eta2,1);
        
        % update phi
        reg = parr*mydiv(eta1,eta2);
        pe=p.*exp(-reg);
        for k=1:K
            temp=pe(:,:,k)./(sum(pe,3)+eps);
            phinew(:,:,k)=temp;
            alpha(k)=sum(temp(:))/(M*N);
        end
        phi = phinew;
        
    end
    
    % according to phi, give a segmentation. 
    [t1 t2]=max(phi,[],3);
    
    % show current segmentation and previous segmentation
    figure(15),
    num1=ceil(K/2);
    for k=1:K
        subplot(num1,2,k),
        imshow(t2==k);
        hold on,
        contour(phiood(:,:,1),[0.5,0.5],'r')
        title(['Class ' num2str(k)]),drawnow;
        imwrite((t2-1)/(K-1),['.\' filename '\' 'iter' num2str(i) 'par=' num2str(parr) 'dt=' num2str(tau) 'inner=' num2str(inner) '.png'])
        
    end
    
end

% Iteration finished, get the final classification
[t1 t2]=max(phi,[],3);
% save img
imwrite((t2-1)/(K-1),['.\' filename '\' 'Final' 'iter' num2str(i) 'par=' num2str(parr) 'dt=' num2str(tau) '.png'])
% show the 3d plot of all points. 
show3dspread(Ima,t2,2)
% show current segmentation
figure(103),imshow(Ima),hold on, contour(t2,[2,2],'r')
