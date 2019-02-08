
%====================================================================
% 3D SAR Based on Measured Data
%--------------------------------
% Step 2: Forming image
%--------------------------------
% INPUT:  dat_expt.mat (from read_json_1pair_HL.m)
% OUTPUT: dat2_expt.mat
%====================================================================

clc;
close all;
clear all;

%======== Data parameters ==========
c = 3e8;        % speed of light
fst = 6.3e9;      % start frequency
fed = 8e9;      % end frequency
fc = 7.15e9;     % center frequency 7.15 GHz
B = fed-fst;    % signal bandwidth: 1.7 GHz
dr = c/B/2;     % range resolution: 0.088m = 8.8cm
Np = 441;        % number of positions
Nf = 4096;       % number of frequency steps

%======== range profile =========
load dat_expt;

NpRange = [];
Atar_2D = reshape(Atar_new,Nf,21,21);
NpRange = Atar_2D(:,:,11);
NpRange = reshape(NpRange,Nf,21);
figure
imagesc(1:21,T(:,1)*3e8/2,abs(NpRange));
xlabel('no. of meas in one direction'); ylabel('depth(m)')
axis([1 21 0 6])
set(gca,'XTick',[1:21],'YTick',[0:6]);

%========== Imaging ================
x_radar = [5:25]*0.01;       % radar positions in x direction
y_radar = [20:40]*0.01;       % radar positions in y direction
d_T2R = 0.02;                  % separation between TX and RX
x_radar_T = x_radar + d_T2R/2;     % TX positions in x direction
x_radar_R = x_radar - d_T2R/2;     % RX positions in x direction
y_radar_T = y_radar + 0;  % TX positions in y direction
y_radar_R = y_radar - 0;  % RX positions in y direction

echo = reshape(Atar_new,Nf,21,21);            % received echoes
F = echo;
NN = 1;
zaxis = (T(:,1)*c/2).';
MFout=20*log10(abs(F(:,11,11))./max(max(max((abs(F))))));
MFout2=20*log10(abs(F(:,21,21))./max(max(max(abs(F)))));

fh=figure
% subplot(2,1,1)
plot(zaxis,MFout,'b-'); hold on
plot(zaxis,MFout2,'g-');
grid on
axis([0 2 -60 0])
xlabel('Range (m)','Fontsize',12)
ylabel('MF Output (dBm)','Fontsize',12)
title('Comparison of signal at target position and at non-target position','Fontsize',12)
legend('signal at target position','signal at other position')
% set(gca,'XTick',[0  3.06 3.825 3.93 4.04 4.1 6],'YTick',[-140 -60])
% subplot(2,1,2)
% rc = floor(zaxis.'/dr);
% plot(rc,MFout,'b-'); hold on
% plot(rc,MFout2,'g-');
% grid on
% axis([0 22 -80 0])
% xlabel('range cell','Fontsize',12)
% ylabel('MF Output (dBm)','Fontsize',12)
% title('(b)','Fontsize',12)
% set(gca,'XTick',[0 0.07 2],'YTick',[-140 -60])
set(fh,'color',[1 1 1])
print -dtiff fig_compareSignal.tif


%======= back projection =========
x_a = -0.05:0.01:0.35;          % targeted imaging area size              
y_a = -0.2:0.01:0.6;   	
z_a = -0.4:0.01:0;               % z axis
z_TR = 0.0;                   % m, ground to TX/RX

% pha_factor = j*2*pi*fc/c;
pha_factor = 0;
FixPath = 71;
deltaD = T(2,1)*c/2; % range separation, for select the correct point in the signal

img = zeros(numel(x_a),numel(y_a),numel(z_a));
for zz = 1:numel(z_a)
    for yy = 1:numel(y_a)
        for xx = 1:numel(x_a)
            for pp = 1:numel(x_radar)
                for qq = 1:numel(y_radar)
                    d = sqrt((x_radar_T(pp)-x_a(xx))^2+(y_radar_T(qq)-y_a(yy))^2+(z_TR-z_a(zz))^2)+...
                        sqrt((x_a(xx)-x_radar_R(pp))^2+(y_a(yy)-y_radar_R(qq))^2+(z_TR-z_a(zz))^2);
                    temp(pp,qq) = F(round(NN*(d/2/deltaD+FixPath)),pp,qq)*exp(pha_factor*d); 
%                     d/2/deltaD
                end
            end
            img(xx,yy,zz) = sum(sum(temp));
        end
    end
end

save dat2_expt.mat img Np Nf dr x_a z_a y_a

disp('Run completed')

% disp('Run GPR Step 3')
% GPRd3step3_display
%eof