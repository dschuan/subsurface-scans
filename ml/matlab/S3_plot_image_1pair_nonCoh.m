%====================================================================
% 3D SAR Based on Measured Data
%--------------------------------
% Step 3: Display image
%--------------------------------
% INPUT:  dat2_expt.mat (from form_image_1pair.m)
% OUTPUT: Image plots
%====================================================================

clc;
close all;
clear all;


load dat2_expt.mat

% x_a, y_a, z_a          % targeted imaging area size              

ImgdB = zeros(length(x_a),length(y_a));
hf4 = figure(4)
for zz = 1:length(z_a)
    ImgdB = 20*log10(abs(img(:,:,zz)));
    surf(y_a,x_a,ImgdB);
    hold on
end
hold off
shading interp; lighting phong; camlight left
xlabel('x (m)');
ylabel('y (m)');
colorbar;
set(hf4,'color',[1 1 1])


xslice = [0.3];
yslice = [0.15];
zslice = [-0.16 ];
[X,Y,Z] = meshgrid(y_a,x_a,z_a);

figure
slice(X,Y,Z,20*log10(abs(img(:,:,:))./max(max(max(abs(img))))),xslice,yslice,zslice); shading interp
colormap jet
colorbar
caxis([-40 0])
set(gca,'YDir','normal','Fontsize',14)
set(gca,'XTick',[0.15 xslice 0.45],'YTick',[0 yslice 0.3],'ZTick',[-0.4 zslice ]);
xlabel('y(m)','Fontsize',14)
ylabel('x(m)','Fontsize',14)
zlabel('z(m)','Fontsize',14)
axis([0.15 0.45 0 0.3 -0.4 0])
view(2)
print -dtiff fig_2D.tif


%eof