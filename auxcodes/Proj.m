function [P1,P2]=Proj(P1,P2,lam)
PN=sqrt(P1.^2+P2.^2);
ind=find(PN>lam);
PNmax=PN(ind);
P1(ind)=lam*P1(ind)./PNmax;
P2(ind)=lam*P2(ind)./PNmax;