close all
clear all
%% Synthetic sample
%102!! 110, 50, 40, 25
fs = 256;
p1_all = [];
% for i = 1:3000%[25,30,31, 40, 50, 59, 73, 84, 100, 102, 104,107, 110, 111, 115, 124, 129]
%     ix = sprintf('%d', i);
%     load(['GAN_data/Train_set_RF/pat_102/GAN_seizure_pat_102_GAN_test_' ix '.mat']);
% 
%     signal = GAN_seiz;
% 
%     signal_E1 = signal(1:length(signal)/2);
%     signal_E2 = signal(length(signal)/2+1:length(signal));
% 
%     signal_E1_filt = bandpass(signal_E1, [1,30], 256);
%     signal_E2_filt = bandpass(signal_E2, [1,30], 256);
%     
%     p1 = bandpower(signal_E1_filt,fs,[0.5 7.5]);
%     p1_all = [p1_all; p1];
% end
% 
% [p1_ordered, p1_order] = sort(p1_all, 'descend');
% 
% for i = [1029, 1089, 1267, 1749, 1737]% [2389, 1737, 277, 817, 865, 843, 1749, 1029, 1553, 1267, 1089, 1462, 1079] %p1_order(1:25)'
%     ix = sprintf('%d', i);
%     load(['GAN_data/Train_set_RF/pat_102/GAN_seizure_pat_102_GAN_test_' ix '.mat']);
% 
%     signal = GAN_seiz;
% 
%     signal_E1 = signal(1:length(signal)/2);
%     signal_E2 = signal(length(signal)/2+1:length(signal));
% 
%     fc = [1, 30];
%     [b,a] = butter(6, fc/(fs/2), 'bandpass');
%     
%     signal_E1_filt = filtfilt(b,a,double(signal_E1));%bandpass(signal_E1, [1,30], 256, 'ImpulseResponse', 'iir');
%     signal_E2_filt = filtfilt(b,a,double(signal_E2));%bandpass(signal_E2, [1,30], 256, 'ImpulseResponse', 'iir');
%     
%     
%     h = figure('units','normalized','outerposition',[0 0 1 1]);
%     ax(1) = subplot(2,1,1);
%     plot(1:length(signal_E1_filt),signal_E1_filt);
%     xlim([0 length(signal_E1_filt)])
%     title(['Synthetic Seizure electrode T7F7: ' ix]);
%     ylabel('Amplitude');
%     xlabel('Time [s]');
%     %ylim([-80 120])
%     xticks([0 256 512 768 1024])
%     xticklabels({'0' '1','2','3','4'})
%     yticks([-200 200])
%     ylim([-200 200]);
%     % 
%     ax(3) = subplot(2,1,2);
%     plot(1:length(signal_E2_filt),signal_E2_filt);  
%     title(['Synthetic Seizure electrode T8F8: ' ix]);  
%     ylabel('Amplitude');
%     xlabel('Time [s]');
%     xlim([0 length(signal_E2_filt)]);
%     %ylim([-80 120]);
%     xticks([0 256 512 768 1024]);
%     xticklabels({'0' '1','2','3','4'});
%     yticks([-200 200])
%     ylim([-200 200]);
%     
%      set(h,'Units','Inches');
%      pos = get(h,'Position');
%      set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
%      print(h,['Figures/Synth_4paper' ix],'-dpdf','-r0');
%      save(['Figures/Synth_4paper' ix '.mat'], 'signal_E1_filt', 'signal_E2_filt');
% end

% Frequency plot
% fft_signal_1 = fft(signal_E1);
% L = length(signal_E1);
% Fs = 256;
% P2 = abs(fft_signal_1/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% P1_1 = P1;
% f = Fs*(0:(L/2))/L;
% figure
% plot(f,P1) 
% title('Single-Sided Amplitude Spectrum of Synthetic X(t) E1')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
% ylim([0 20])

    
%% Real sampl
load('pat_114902_seiz.mat');

% for i = [213, 322, 373, 381, 416]%[105, 117]
%     ix = sprintf('%d', i);
%     signal = X_seiz(i,:);
signal  = X_seiz;

    signal_E1 = signal(1:length(signal)/2);
    signal_E2 = signal(length(signal)/2+1:length(signal));

    fc = [1, 30];
    [b,a] = butter(6, fc/(fs/2), 'bandpass');     
    
    signal_E1_filt = filtfilt(b,a,double(signal_E1));%bandpass(signal_E1, [1,30], 256);
    signal_E2_filt = filtfilt(b,a,double(signal_E2));%bandpass(signal_E2, [1,30], 256);

    h = figure('units','normalized','outerposition',[0 0 1 1]);
    ax(1) = subplot(2,1,1);
    plot(1:length(signal_E1_filt),signal_E1_filt);
    xlim([0 length(signal_E1_filt)])
%     title(['Real Seizure electrode T7F7: ']); 
    ylabel('T3F7', 'FontSize',24,'FontWeight','bold');
    xlabel('time (second)', 'FontSize',24,'FontWeight','bold');
    xticks([1, 256, 512, 768, 1024])
    xticklabels({'0', '1','2','3','4'})
    ylim([-200 200]);
    yticks([-200 0 200])
    ax(1).FontSize = 20;

    ax(3) = subplot(2,1,2);
    plot(1:length(signal_E2_filt),signal_E2_filt);  
%     title(['Real Seizure electrode T8F8: ']);  
    ylabel('T4F8','FontSize',24,'FontWeight', 'bold');
    xlabel('time (second)', 'FontSize',24,'FontWeight','bold');
    xlim([0 length(signal_E2_filt)])
    %ylim([-80 120]);
    xticks([1, 256, 512, 768, 1024])
    xticklabels({'0', '1','2','3','4'})
    ylim([-200 200]);
    yticks([-200 0 200])
ax(3).FontSize = 20; 
    
    %ylim([-450 300])
    set(h,'Units','Inches');
    
    pos = get(h,'Position');
    set(h,'PaperPosition',[0 0 9.62 7.4],'PaperUnits','Inches','PaperSize',[9.62 7.4])
    print(h,'2_Real_4paper','-dpdf');
% saveas(h, 'real.pdf')
%     save(['Figures/2_GAN_seizure_pat_102_real_' ix '.mat'], 'signal_E1', 'signal_E2');
% end

%% 
% Frequency plot
% fft_signal_1 = fft(signal_E1);
% L = length(signal_E1);
% Fs = 256;
% P2 = abs(fft_signal_1/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% P1_1 = P1;
% f = Fs*(0:(L/2))/L;
% figure
% plot(f,P1) 
% title('Single-Sided Amplitude Spectrum of Real X(t) E1')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
% ylim([0 20])
