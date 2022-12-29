clc
clear;
close all
load alldatan.mat

Ima = imread('alldone_2.png');
[maxv,segres]=max(phi,[],3);
seg = segres;
%Ima;

figure(1),iptsetpref('ImshowBorder','tight');imshow(Ima),hold on,contour(seg,[1,1],'r','LineWidth',6)
print -f1 -depsc -tiff -r600 NPMM_result
show3dspread_ori(Ima,seg,2)