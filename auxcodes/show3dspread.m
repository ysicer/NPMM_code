function show3dspread(Ima,t2,part)
[row,col,dim] = size(Ima);
Ima31 = reshape(Ima(:,:,1),[row*col,1]);
Ima32 = reshape(Ima(:,:,2),[row*col,1]);
Ima33 = reshape(Ima(:,:,3),[row*col,1]);
figure(11),plot3(Ima31(:),Ima32(:),Ima33(:),'+'),title('Original distribution'),drawnow;

if part==2
    t22 = reshape(t2,[row*col,1]);
    try
        close(120)
    end
    figure(120),
    %iptsetpref('ImshowBorder','tight')
    plot3(Ima31(t22==2),Ima32(t22==2),Ima33(t22==2),'r.')
    hold on
    plot3(Ima31(t22==1),Ima32(t22==1),Ima33(t22==1),'g.')
    axis on
    set(gca,'XTick',[],'YTick',[],'ZTick',[]);
    view(-60,45)
    title('Segmentation distribution')
    %,drawnow;
    %print -f12 -depsc -tiff -r600 NPMM_spatial
end
