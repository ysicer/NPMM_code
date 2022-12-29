function divP=mydiv(P1,P2)


P1x1=P1(2:end-1,:,:)-P1(1:end-2,:,:);
P1x=cat(1,P1(1,:,:),P1x1,-P1(end-1,:,:));

P2y2=P2(:,2:end-1,:)-P2(:,1:end-2,:);
P2y=cat(2,P2(:,1,:),P2y2,-P2(:,end-1,:));
divP=P1x+P2y;

%{
[P1x P1y]=gradient(P1);
[P2x P2y]=gradient(P2);
divP=P1x+P2y;
%}
%{
P1E=padarray(P1,[2 2],'both','symmetric');
P2E=padarray(P2,[2 2],'both','symmetric');
P1x1=P1E(4:end-1,3:end-2,:)-P1E(2:end-3,3:end-2,:);
P1x2=P1E(5:end,3:end-2,:)-P1E(1:end-4,3:end-2,:);
P1x=P1x1/4+P1x2/8;size(P1x)
P2y1=P2E(3:end-2,4:end-1,:)-P2E(3:end-2,2:end-3,:);
P2y2=P2E(3:end-2,5:end,:)-P2E(3:end-2,1:end-4,:);
P2y=P2y1/4+P2y2/8;size(P2y)
divP=P1x+P2y;
%}