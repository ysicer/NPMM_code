function [phinew,alpha,p]=NonPara3dEStep_smooth(I,phi,reg,alpha)
% to calculate the NPMM probility map. 
[M N K]=size(phi);

for k=1:K
    p1=DensityFun_3d_2(I(:,:,1),I(:,:,2),I(:,:,3),phi(:,:,k),1e-1);
    p(:,:,k)=p1*alpha(k);
end
pe=p.*exp(-reg);
for k=1:K
    temp=pe(:,:,k)./(sum(pe,3)+eps);
    phinew(:,:,k)=temp;
    alpha(k)=sum(temp(:))/(M*N);
end

