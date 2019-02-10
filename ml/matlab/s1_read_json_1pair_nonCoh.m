clear all; close all

fname = '07_02Aluminum.json';
% fname = '07_02&AlumWithCover.json';
val = jsondecode(fileread(fname));          % worked
fn = fieldnames(val);
expt = val.(fn{1});
Nm = length(expt);
for ii = 1:Nm
    expt2 = expt(ii);
    coord = expt2.coord
    body = expt2.body;
    A(:,ii) = body.amplitude;
    T(:,ii) = body.time;
end

%plot(amplitude, time)
% filt = [zeros(1,144) ones(1,length(A(:,1))-144)];
% A_filt = A.*(filt.'*ones(1,Nm));
Aabs=abs(A); Amax=max(max(Aabs)); AdB=20*log10(Aabs/Amax);
T1ns=T(:,1)*1e9; % time in nsdep
Amax_2D = reshape(max(abs(A)),21,21);
figure
imagesc(20*log10(Amax_2D./max(max(Amax_2D)))); 
xlabel('y'); ylabel('x'); title('max value(dB) of raw signal')
axis equal
view(2)
print -dtiff fig_maxValue.tif

figure
plot(T1ns*3e-1/2,A(:,11),'b--','linewidth',2); hold on
plot(T1ns*3e-1/2,A(:,6),'g-','linewidth',2);
plot(T1ns*3e-1/2,A(:,2),'m-.','linewidth',2);
plot(T1ns*3e-1/2,A(:,1),'c:','linewidth',2); 
hold off
xlabel('depth (m)'); ylabel('Amplitude');
axis([0 2 -0.8 0.8])
legend('11th','6th','2th','1th')
print -dtiff fig_exampleSig.tif

% convert to frequency domain
N = length(A(:,1)); Tmax = T(N,1); fs = N/Tmax;
faxis = ([0:1/N:(1-1/N)])*fs;
Afft = (fft(A)); %4096x441
Afft_dB = 20*log10(abs(Afft)./(max(max(Afft))));

% convert to complex signal
Afft_new = [Afft(1:N/2,:);zeros(N/2,Nm)];
A_new = ifft(Afft_new);
figure
plot(T1ns*3e-1/2,abs(A_new));
xlabel('depth (m)'); ylabel('abs');
axis([0 2 0 0.4])
print -dtiff fig_complexSignal_Envelope.tif

AdB_new = 20*log10(abs(A_new)./max(max(abs(A_new))));
figure
surf(1:N,[1:Nm],AdB_new.')
xlabel('Range cell'); ylabel('Measurement idx'); zlabel('Amplitude (dB)')
title('Signal from all measurements')
zlim([-100 0])
shading interp
view(2)

%=== remove background signal: direct pulse, scattering from others, etc ==
fname3 = '07_02No_Target.json'; % read background data
val3 = jsondecode(fileread(fname3));   % worked
fn3 = fieldnames(val3);
expt3 = val3.(fn3{1});
Nm = length(expt3);
for ii = 1:Nm
    expt4 = expt3(ii);
    coord4 = expt4.coord
    body4 = expt4.body;
    A2(:,ii) = body4.amplitude; % background data
    T2(:,ii) = body4.time;
end
bg = sum(A2,2)./Nm; % averaged background signal

A_tar = A-bg*ones(1,Nm); % target signal with background removed
figure
plot(T1ns*3e-1/2,A_tar(:,11),'b--','linewidth',2); hold on
plot(T1ns*3e-1/2,A_tar(:,6),'g-','linewidth',2);
plot(T1ns*3e-1/2,A_tar(:,2),'m-.','linewidth',2);
plot(T1ns*3e-1/2,A_tar(:,1),'c:','linewidth',2); 
hold off
xlabel('depth (m)'); ylabel('Amplitude');
title('Target signal with background removal')
axis([0 2 -0.8 0.8])
legend('11th','6th','2th','1th')
print -dtiff fig_exampleSig_afterBGremove.tif

Afft2 = fft(A_tar); %4096x441
Afft2_dB = 20*log10(abs(Afft2)./(max(max(Afft2))));
Afft2_new = [Afft2(1:N/2,:);zeros(N/2,Nm)];
Atar_new = ifft(Afft2_new); % convert to complex number
figure
plot(T1ns*3e-1/2,abs(Atar_new));
xlabel('depth (m)'); ylabel('abs');
axis([0 2 0 0.4])
title('new signal without background')
print -dtiff fig_newComplexSig_noBG.tif


AdB_new2 = 20*log10(abs(Atar_new)./max(max(abs(Atar_new))));
figure
surf(T1ns*3e-1/2,[1:Nm],AdB_new2.')
xlabel('Range (m)'); ylabel('Measurement idx'); zlabel('Amplitude (dB)')
title('New signal from all measurements')
zlim([-100 0])
shading interp
view(2)

Amax_2D = reshape(max(abs(Atar_new)),21,21);
figure
surfc(20*log10(Amax_2D./max(max(Amax_2D)))); colorbar % shading flat 
xlabel('y'); ylabel('x'); title('max value(dB) of new signal after bg removal')
axis equal
view(2)
print -dtiff fig_maxVlaue_noBG.tif


save dat_expt.mat T A_tar Atar_new

%====== plot demo figures ======
Atar_2D = reshape(Atar_new,N,21,21);
% Atar_2D_sum = sum(Atar_2D,[2]);
Atar_2D_sum = Atar_2D(:,10,:);
Atar_2D_sum2 = reshape(Atar_2D_sum,N,21);
figure
imagesc(1:21,T1ns*3e-1/2,abs(Atar_2D_sum2));
xlabel('no. of meas in y direction')
ylabel('depth(m)')
axis([1 21 0 2])
print -dtiff fig_1Dmeas_y.tif


Atar_2D_sum3 = Atar_2D(:,:,10);
Atar_2D_sum4 = reshape(Atar_2D_sum3,N,21);
figure
imagesc(1:21,T1ns*3e-1/2,abs(Atar_2D_sum4));
xlabel('no. of meas in x direction')
ylabel('depth(m)')
axis([1 21 0 2])
print -dtiff fig_1Dmeas_x.tif

Afft3 = fft(A2); %4096x441
Afft3_new = [Afft3(1:N/2,:);zeros(N/2,Nm)];
A2_new = ifft(Afft3_new); % convert to complex number
figure
plot(T1ns*3e-1/2,abs(A_new(:,221)),'r','linewidth',2); hold on
plot(T1ns*3e-1/2,abs(A2_new(:,221)),'b--','linewidth',2); hold off
xlabel('depth (m)'); ylabel('Amplitude');
title('Compare signal')
axis([0 2 0 0.4])
legend('with target','without target')
print -dtiff fig_compareWtTarWoTar.tif


figure
xx=[5:1:25]; Nx=length(xx);
yy=[20:1:40]; Ny=length(yy);
Nt=Nx*Ny;
%[X,Y] = meshgrid(xx,yy);
x4=[xx(1) xx(Nx) xx(Nx) xx(1)];
y4=[yy(1) yy(1) yy(Ny) yy(Ny)];
fill(x4,y4,[1.0 0.8 0.6]); hold on
np=1;
for n=1:Ny
    for m=1:Nx
        text(xx(m)-0.3,yy(n)+0.1,num2str(np))
        np=np+1;
    end
end
hold off
axis([3 27 18 42])
% axis equal
xlabel('x cm')
ylabel('y cm')
set(gca,'Xtick',xx,'Ytick',yy)
grid
print -dtiff fig_scan.tif

% save all_data.mat 

