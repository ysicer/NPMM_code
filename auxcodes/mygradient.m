function [Ux,Uy]=mygradient(U)

[M N K]=size(U);
Ux1=U(2:end,:,:)-U(1:end-1,:,:);
Ux=cat(1,Ux1,zeros(1,N,K));

Uy1=U(:,2:end,:)-U(:,1:end-1,:);
Uy=cat(2,Uy1,zeros(M,1,K));


%[Ux,Uy]=gradient(U);

%{
UE=padarray(U,[2 2],'both','symmetric');
Ux1=UE(4:end-1,3:end-2,:)-UE(2:end-3,3:end-2,:);
Ux2=UE(5:end,3:end-2,:)-UE(1:end-4,3:end-2,:);
Ux=Ux1/4+Ux2/8;
Uy1=UE(3:end-2,4:end-1,:)-UE(3:end-2,2:end-3,:);
Uy2=UE(3:end-2,5:end,:)-UE(3:end-2,1:end-4,:);
Uy=Uy1/4+Uy2/8;
%}