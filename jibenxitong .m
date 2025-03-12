% clc;
% clear all;
% 
% %% 初始化参数设置
% rolloff_factor = 0.3;    % 调整滚降因子
% fs = 48000;              % 采样频率，确保在其他地方定义
% fc = 1800;
% T_s = 1/fs;              % 信道采样间隔
% condition = 1;           % 信道条件
% N = 10000;                % 符号数
% sps1 = 40;               % 第一次上采样倍数
% 
% 
% %% 生成载波
% t=0:1./fs:50; 
% CarryFreq=exp(1i*2.*pi*fc*t);
% 
% %% 产生随机二进制01信息
% x_bin = randi([0 1], 1, N); % 生成1*N的行向量
% 
% %% 调制
% x_t = BPSK_Modulate(x_bin);
% 
% %% 第一次上采样（40倍）
% x_t_upsampled1 = upsample(x_t, sps1);
% 
% %% 成型滤波
% rcos_fir = rcosdesign(rolloff_factor, 8, sps1, 'sqrt');
% shaped_bpsk_message1 = conv(x_t_upsampled1, rcos_fir, 'same');
% 
% %% 乘载波
% tx_signal=shaped_bpsk_message1.*CarryFreq(1:length(shaped_bpsk_message1));
% 
% %% 通过信道传输
% y_nbhf = NBHF_channel(tx_signal, T_s, condition); % 使用定义的信道函数
% 
% %% 设置不同信噪比范围
% SNR_dB = 10:1:30; % 设置信噪比范围和步长
% ber = zeros(1, length(SNR_dB)); % 初始化误比特率数组
% 
% for i = 1:length(SNR_dB)
%     %% 添加高斯白噪声
%     y_noise = awgn(y_nbhf, SNR_dB(i), 'measured'); % 添加高斯白噪声
% 
%     %% 下变频
%     y_noise=y_noise.*conj(CarryFreq(1:length(shaped_bpsk_message1)));
% 
%     %% 匹配滤波
%     rx_signal = conv(y_noise, rcos_fir, 'same');
% 
%     %% 下采样
%     downsampled_rx_signal = rx_signal(1:sps1:end); % 对信号进行下采样
% 
%     %% BPSK解调
%     rx_data = BPSK_Demodulate(downsampled_rx_signal); % 进行BPSK解调
% 
%     %% 计算误比特率
%     x_bin = x_bin(1:length(rx_data)); % 截断原始数据长度以匹配接收数据
%     ber(i) = sum(abs(x_bin - rx_data)) / length(x_bin); % 计算误比特率
% 
%     %% 打印当前信噪比下的误比特率
%     fprintf('SNR = %d dB, BER = %e\n', SNR_dB(i), ber(i));
% end
% 
% save("chuantongxitongSNR.mat","SNR_dB","ber");
% 
% %% 绘制误比特率曲线
% figure;
% plot(SNR_dB, ber, 'o-');
% xlabel('SNR (dB)');
% ylabel('BER');
% title('BER vs. SNR');
% grid on;
% 
% %% 绘制原始信源数据和解调后的数据的对比图
% figure;
% subplot(2, 1, 1);
% plot(1:length(x_bin), x_bin, 'b', 'LineWidth', 1.5);
% title('原始信源数据');
% xlabel('样本点');
% ylim([-0.5 1.5]);
% 
% subplot(2, 1, 2);
% plot(1:length(rx_data), rx_data, 'r', 'LineWidth', 1.5);
% title('解调后的数据');
% xlabel('样本点');
% ylabel('二进制数据');
% ylim([-0.5 1.5]);


% clc;
% clear;
% 
% %% 初始化参数设置
% rolloff_factor = 0.3;    % 调整滚降因子
% fs = 48000;              % 采样频率
% fc = 1800;               % 载波频率
% T_s = 1/fs;              % 信道采样间隔
% condition = 2;           % 信道条件
% N = 1008;                % 符号数
% sps1 = 40;               % 第一次上采样倍数
% 
% %% 生成载波
% t = 0:1/fs:10;
% CarryFreq = exp(1i*2*pi*fc*t);
% 
% %% 设置信噪比范围
% SNR_dB = -10:1:10; % 设置信噪比范围和步长
% ber = zeros(1, length(SNR_dB)); % 初始化误比特率数组
% total_time = 1; % 总时间为30秒
% 
% %% 30秒的循环
% for second = 1:total_time
%     %% 产生随机二进制01信息
%     x_bin = randi([0 1], 1, N); 
% 
%     %% 调制
%     x_t = BPSK_Modulate(x_bin);
% 
%     %% 第一次上采样（40倍）
%     x_t_upsampled1 = upsample(x_t, sps1);
% 
%     %% 成型滤波
%     rcos_fir = rcosdesign(rolloff_factor, 8, sps1, 'sqrt');
%     shaped_qam4_message1 = conv(x_t_upsampled1, rcos_fir, 'same');
% 
%     %% 乘载波
%     tx_signal = shaped_qam4_message1 .* CarryFreq(1:length(shaped_qam4_message1));
% 
%     %% 通过信道传输
%     y_nbhf = NBHF_channel(tx_signal, T_s, condition); % 使用定义的信道函数
% 
%     for i = 1:length(SNR_dB)
%         %% 添加高斯白噪声
%         y_noise = awgn(y_nbhf, SNR_dB(i), 'measured'); % 添加高斯白噪声
% 
%         %% 下变频
%         y_noise = y_noise .* conj(CarryFreq(1:length(shaped_qam4_message1)));
% 
%         %% 匹配滤波
%         rx_signal = conv(y_noise, rcos_fir, 'same');
% 
%         %% 下采样
%         downsampled_rx_signal = rx_signal(1:sps1:end); % 对信号进行下采样
% 
%         %% 解调
%         rx_data = BPSK_Demodulate(downsampled_rx_signal); 
% 
%         %% 计算误比特率并累积
%         x_bin = x_bin(1:length(rx_data)); % 截断原始数据长度以匹配接收数据
% 
%         ber(i) = ber(i) + sum(abs(x_bin - rx_data)) / length(x_bin); % 累积误比特率
% 
%         %% 打印当前秒数和信噪比下的误比特率
%         fprintf('Second = %d, SNR = %d dB, BER = %e\n', second, SNR_dB(i), ber(i)/second);
%     end
% end
% 
% %% 计算平均误比特率
% ber = ber / total_time;
% 
% save("chuantongxitong.mat","SNR_dB","ber");
% 
% %% 绘制平均误比特率曲线
% figure;
% plot(SNR_dB, ber, 'o-');
% xlabel('SNR (dB)');
% ylabel('平均BER');
% title('平均BER vs. SNR');
% grid on;


































% clc;
% clear all;
% 
% %% 初始化参数设置
% rolloff_factor = 0.5;    % 滚降因子
% fs = 48000;              % 采样频率
% fc = 1800;               % 载波频率
% T_s = 1/fs;              % 信道采样间隔
% N = 100000;                % 符号数
% sps1 = 8;                % 第一次上采样倍数
% sps2 = 20;               % 第二次上采样后的总倍数
% 
% %% 生成信源数据
% x_bin = randi([0, 1], 1, N); % 生成随机二进制序列
% 
% %% BPSK调制
% x_t = BPSK_Modulate(x_bin); % 将二进制数据进行BPSK调制
% 
% %% 第一次上采样（8倍）
% x_t_upsampled1 = upsample(x_t, sps1); % 进行第一次上采样
% 
% %% 成型滤波
% rcos_fir = rcosdesign(rolloff_factor, 8, sps1, 'sqrt'); % 生成根升余弦滤波器
% shaped_bpsk_message1 = conv(x_t_upsampled1, rcos_fir, 'same'); % 进行成型滤波
% 
% %% 第二次上采样（在8倍上采样的基础上进行2.5倍上采样）
% shaped_bpsk_message2 = resample(shaped_bpsk_message1, sps2, sps1); % 进行第二次上采样
% 
% %% 计算相位增量和相位跳变
% phase_step = 2 * pi * fc / fs; % 相位步长
% phase_jump = 2 * pi - (20 * phase_step); % 相位跳变
% 
% %% 生成相位累加器的载波信号并进行点乘
% N_samples = length(shaped_bpsk_message2);
% phase_accumulator = 0;  % 初始化相位累加器
% 
% tx_signal = zeros(1, N_samples); % 初始化调制信号
% phase_values = zeros(1, N_samples); % 用于存储相位值以便绘图
% for i = 1:N_samples
%     carrier = cos(phase_accumulator); % 生成载波信号
%     tx_signal(i) = shaped_bpsk_message2(i) * carrier; % 将信号逐点乘上载波
%     phase_values(i) = phase_accumulator; % 存储当前相位值
%     phase_accumulator = phase_accumulator + phase_step; % 更新相位累加器
%     if mod(i, 20) == 0
%         phase_accumulator = phase_accumulator + phase_jump; % 修正相位跳变
%     end
% end
% 
% 
% %% 设置不同信噪比范围
% SNR_dB = -10:1:20; % 设置信噪比范围和步长
% ber = zeros(1, length(SNR_dB)); % 初始化误比特率数组
% 
% for i = 1:length(SNR_dB)
%     %% 添加高斯白噪声
%     y_filtered = awgn(tx_signal, SNR_dB(i), 'measured'); % 添加高斯白噪声
% 
%     % % 下变频
%     % y_filtered = y_filtered .* carrier; % 将接收信号乘以载波进行下变频
% 
%     % %% 低通滤波器，用于滤除高频和镜像频率成分
%     % lpFilt = designfilt('lowpassfir', 'PassbandFrequency', 0.3, ...
%     %     'StopbandFrequency', 0.4, 'PassbandRipple', 0.01, ...
%     %     'StopbandAttenuation', 60, 'DesignMethod', 'kaiserwin'); % 设计低通滤波器
%     % 
%     % %% 使用滤波器系数进行卷积
%     % rx_signal_filtered = conv(y_filtered, lpFilt.Coefficients, 'same'); % 对信号进行低通滤波
% 
%     %% 匹配滤波
%     rx_signal = conv(y_filtered, rcos_fir, 'same'); % 进行匹配滤波
% 
%     %% 下采样
%     downsampled_rx_signal = rx_signal(1:sps2:end); % 对信号进行下采样
% 
%     %% BPSK解调
%     rx_data = BPSK_Demodulate(downsampled_rx_signal); % 进行BPSK解调
% 
%     %% 计算误比特率
%     x_bin = x_bin(1:length(rx_data)); % 截断原始数据长度以匹配接收数据
%     ber(i) = sum(abs(x_bin - rx_data)) / length(x_bin); % 计算误比特率
% 
%     %% 打印当前信噪比下的误比特率
%     fprintf('SNR = %d dB, BER = %e\n', SNR_dB(i), ber(i));
% end
% 
% %% 绘制误比特率曲线
% figure;
% plot(SNR_dB, ber, 'o-');
% xlabel('SNR (dB)');
% ylabel('BER');
% title('BER vs. SNR');
% grid on;
% 
% %% 绘制原始信源数据和解调后的数据的对比图
% figure;
% subplot(2, 1, 1);
% plot(1:length(x_bin), x_bin, 'b', 'LineWidth', 1.5);
% title('原始信源数据');
% xlabel('样本点');
% ylim([-0.5 1.5]);
% 
% subplot(2, 1, 2);
% plot(1:length(rx_data), rx_data, 'r', 'LineWidth', 1.5);
% title('解调后的数据');
% xlabel('样本点');
% ylabel('二进制数据');
% ylim([-0.5 1.5]);






% % % % clc;
% % % % clear all;
% % % % 
% % % % %% 初始化参数设置
% % % % rolloff_factor = 0.5;    % 滚降因子
% % % % fs = 48000;              % 采样频率
% % % % fc = 1800;               % 载波频率
% % % % T_s = 1/fs;              % 信道采样间隔
% % % % condition = 2;           % 信道条件
% % % % N = 10000;               % 符号数
% % % % sps1 = 8;                % 第一次上采样倍数
% % % % sps2 = 20;               % 第二次上采样后的总倍数
% % % % 
% % % % %% 生成信源数据
% % % % x_bin = randi([0, 1], 1, N); % 生成随机二进制序列
% % % % 
% % % % %% 绘制x_bin的波形
% % % % figure;
% % % % subplot(2, 1, 1);
% % % % plot(1:length(x_bin), x_bin, 'b', 'LineWidth', 1.5);
% % % % title('原始二进制序列 x\_bin');
% % % % xlabel('样本点');
% % % % ylabel('幅度');
% % % % ylim([-0.5 1.5]);
% % % % grid on;
% % % % 
% % % % %% 计算并绘制x_bin的双边频谱图
% % % % N_fft = length(x_bin);
% % % % X_bin_fft = fft(x_bin);
% % % % f_bin = (-N_fft/2:N_fft/2-1)*(fs/N_fft);
% % % % 
% % % % subplot(2, 1, 2);
% % % % plot(f_bin, fftshift(abs(X_bin_fft)), 'r', 'LineWidth', 1.5);
% % % % title('x\_bin 的双边频谱图');
% % % % xlabel('频率 (Hz)');
% % % % ylabel('幅度');
% % % % grid on;
% % % % 
% % % % %% BPSK调制
% % % % x_t = BPSK_Modulate(x_bin); % 将二进制数据进行BPSK调制
% % % % 
% % % % %% 绘制x_t的波形和频谱图
% % % % figure;
% % % % subplot(2, 1, 1);
% % % % plot(1:length(x_t), x_t, 'b', 'LineWidth', 1.5);
% % % % title('BPSK调制信号 x\_t');
% % % % xlabel('样本点');
% % % % ylabel('幅度');
% % % % grid on;
% % % % 
% % % % N_fft = length(x_t);
% % % % X_t_fft = fft(x_t);
% % % % f_t = (-N_fft/2:N_fft/2-1)*(fs/N_fft);
% % % % 
% % % % subplot(2, 1, 2);
% % % % plot(f_t, fftshift(abs(X_t_fft)), 'r', 'LineWidth', 1.5);
% % % % title('BPSK调制信号 x\_t 的双边频谱图');
% % % % xlabel('频率 (Hz)');
% % % % ylabel('幅度');
% % % % grid on;
% % % % 
% % % % %% 第一次上采样（8倍）
% % % % x_t_upsampled1 = upsample(x_t, sps1); % 进行第一次上采样
% % % % 
% % % % %% 成型滤波
% % % % rcos_fir = rcosdesign(rolloff_factor, 8, sps1, 'sqrt'); % 生成根升余弦滤波器
% % % % shaped_bpsk_message1 = conv(x_t_upsampled1, rcos_fir, 'same'); % 进行成型滤波
% % % % 
% % % % %% 绘制shaped_bpsk_message1的波形和频谱图
% % % % figure;
% % % % subplot(2, 1, 1);
% % % % plot(1:length(shaped_bpsk_message1), shaped_bpsk_message1, 'b', 'LineWidth', 1.5);
% % % % title('成型滤波后的信号 shaped\_bpsk\_message1');
% % % % xlabel('样本点');
% % % % ylabel('幅度');
% % % % grid on;
% % % % 
% % % % N_fft = length(shaped_bpsk_message1);
% % % % Shaped_bpsk_message1_fft = fft(shaped_bpsk_message1);
% % % % f_shaped1 = (-N_fft/2:N_fft/2-1)*(fs/N_fft);
% % % % 
% % % % subplot(2, 1, 2);
% % % % plot(f_shaped1, fftshift(abs(Shaped_bpsk_message1_fft)), 'r', 'LineWidth', 1.5);
% % % % title('成型滤波后的信号 shaped\_bpsk\_message1 的双边频谱图');
% % % % xlabel('频率 (Hz)');
% % % % ylabel('幅度');
% % % % grid on;
% % % % 
% % % % %% 第二次上采样（在8倍上采样的基础上进行2.5倍上采样）
% % % % shaped_bpsk_message2 = resample(shaped_bpsk_message1, sps2, sps1); % 进行第二次上采样
% % % % 
% % % % %% 绘制shaped_bpsk_message2的波形和频谱图
% % % % figure;
% % % % subplot(2, 1, 1);
% % % % plot(1:length(shaped_bpsk_message2), shaped_bpsk_message2, 'b', 'LineWidth', 1.5);
% % % % title('第二次上采样后的信号 shaped\_bpsk\_message2');
% % % % xlabel('样本点');
% % % % ylabel('幅度');
% % % % grid on;
% % % % 
% % % % N_fft = length(shaped_bpsk_message2);
% % % % Shaped_bpsk_message2_fft = fft(shaped_bpsk_message2);
% % % % f_shaped2 = (-N_fft/2:N_fft/2-1)*(fs/N_fft);
% % % % 
% % % % subplot(2, 1, 2);
% % % % plot(f_shaped2, fftshift(abs(Shaped_bpsk_message2_fft)), 'r', 'LineWidth', 1.5);
% % % % title('第二次上采样后的信号 shaped\_bpsk\_message2 的双边频谱图');
% % % % xlabel('频率 (Hz)');
% % % % ylabel('幅度');
% % % % grid on;
% % % % 
% % % % %% 生成载波信号
% % % % t = (0:length(shaped_bpsk_message2)-1) / fs; % 生成时间向量
% % % % carrier = cos(2 * pi * fc * t); % 生成载波信号
% % % % 
% % % % %% 将信号乘上载波
% % % % tx_signal = shaped_bpsk_message2 .* carrier; % 调制信号乘以载波
% % % % 
% % % % %% 绘制tx_signal的波形和频谱图
% % % % figure;
% % % % subplot(2, 1, 1);
% % % % plot(1:length(tx_signal), tx_signal, 'b', 'LineWidth', 1.5);
% % % % title('调制后的传输信号 tx\_signal');
% % % % xlabel('样本点');
% % % % ylabel('幅度');
% % % % grid on;
% % % % 
% % % % N_fft = length(tx_signal);
% % % % Tx_signal_fft = fft(tx_signal);
% % % % f_tx = (-N_fft/2:N_fft/2-1)*(fs/N_fft);
% % % % 
% % % % subplot(2, 1, 2);
% % % % plot(f_tx, fftshift(abs(Tx_signal_fft)), 'r', 'LineWidth', 1.5);
% % % % title('调制后的传输信号 tx\_signal 的双边频谱图');
% % % % xlabel('频率 (Hz)');
% % % % ylabel('幅度');
% % % % grid on;
% % % % 
% % % % %% NBHF信道传输
% % % % y_nbhf = NBHF_channel(tx_signal, T_s, condition); % 通过NBHF信道传输
% % % % 
% % % % %% 绘制y_nbhf的波形和频谱图
% % % % figure;
% % % % subplot(2, 1, 1);
% % % % plot(1:length(y_nbhf), y_nbhf, 'b', 'LineWidth', 1.5);
% % % % title('NBHF信道输出信号 y\_nbhf');
% % % % xlabel('样本点');
% % % % ylabel('幅度');
% % % % grid on;
% % % % 
% % % % N_fft = length(y_nbhf);
% % % % Y_nbhf_fft = fft(y_nbhf);
% % % % f_nbhf = (-N_fft/2:N_fft/2-1)*(fs/N_fft);
% % % % 
% % % % subplot(2, 1, 2);
% % % % plot(f_nbhf, fftshift(abs(Y_nbhf_fft)), 'r', 'LineWidth', 1.5);
% % % % title('NBHF信道输出信号 y\_nbhf 的双边频谱图');
% % % % xlabel('频率 (Hz)');
% % % % ylabel('幅度');
% % % % grid on;
% % % % 
% % % % %% 设置不同信噪比范围
% % % % SNR_dB = 10:1:30; % 设置信噪比范围和步长
% % % % ber = zeros(1, length(SNR_dB)); % 初始化误比特率数组
% % % % 
% % % % for i = 1:length(SNR_dB)
% % % %     %% 添加高斯白噪声
% % % %     y_filtered = awgn(y_nbhf, SNR_dB(i), 'measured'); % 添加高斯白噪声
% % % % 
% % % %     %% 下变频
% % % %     y_filtered = y_filtered .* carrier; % 将接收信号乘以载波进行下变频
% % % % 
% % % %     %% 绘制rx_signal_filtered的波形和频谱图
% % % %     figure;
% % % %     subplot(2, 1, 1);
% % % %     plot(1:length(y_filtered), y_filtered, 'b', 'LineWidth', 1.5);
% % % %     title('低通滤波后的信号y_filtered');
% % % %     xlabel('样本点');
% % % %     ylabel('幅度');
% % % %     grid on;
% % % % 
% % % %     N_fft = length(y_filtered);
% % % %     y_filtered_fft = fft(y_filtered);
% % % %     y_rx_filtered = (-N_fft/2:N_fft/2-1)*(fs/N_fft);
% % % % 
% % % %     subplot(2, 1, 2);
% % % %     plot(y_rx_filtered, fftshift(abs(y_filtered_fft)), 'r', 'LineWidth', 1.5);
% % % %     title('低通滤波后的信号 y_filtered 的双边频谱图');
% % % %     xlabel('频率 (Hz)');
% % % %     ylabel('幅度');
% % % %     grid on;
% % % % 
% % % %     %% 低通滤波器，用于滤除高频和镜像频率成分
% % % %     lpFilt = designfilt('lowpassfir', 'PassbandFrequency', 0.2, ...
% % % %         'StopbandFrequency', 0.3, 'PassbandRipple', 0.01, ...
% % % %         'StopbandAttenuation', 60, 'DesignMethod', 'kaiserwin'); % 设计低通滤波器
% % % % 
% % % %     %% 使用滤波器系数进行卷积
% % % %     rx_signal_filtered = conv(y_filtered, lpFilt.Coefficients, 'same'); % 对信号进行低通滤波
% % % % 
% % % %     %% 绘制rx_signal_filtered的波形和频谱图
% % % %     figure;
% % % %     subplot(2, 1, 1);
% % % %     plot(1:length(rx_signal_filtered), rx_signal_filtered, 'b', 'LineWidth', 1.5);
% % % %     title('低通滤波后的信号 rx\_signal\_filtered');
% % % %     xlabel('样本点');
% % % %     ylabel('幅度');
% % % %     grid on;
% % % % 
% % % %     N_fft = length(rx_signal_filtered);
% % % %     Rx_signal_filtered_fft = fft(rx_signal_filtered);
% % % %     f_rx_filtered = (-N_fft/2:N_fft/2-1)*(fs/N_fft);
% % % % 
% % % %     subplot(2, 1, 2);
% % % %     plot(f_rx_filtered, fftshift(abs(Rx_signal_filtered_fft)), 'r', 'LineWidth', 1.5);
% % % %     title('低通滤波后的信号 rx\_signal\_filtered 的双边频谱图');
% % % %     xlabel('频率 (Hz)');
% % % %     ylabel('幅度');
% % % %     grid on;
% % % % 
% % % %     %% 匹配滤波
% % % %     rx_signal = conv(rx_signal_filtered, rcos_fir, 'same'); % 进行匹配滤波
% % % % 
% % % %     %% 绘制rx_signal的波形和频谱图
% % % %     figure;
% % % %     subplot(2, 1, 1);
% % % %     plot(1:length(rx_signal), rx_signal, 'b', 'LineWidth', 1.5);
% % % %     title('匹配滤波后的信号 rx\_signal');
% % % %     xlabel('样本点');
% % % %     ylabel('幅度');
% % % %     grid on;
% % % % 
% % % %     N_fft = length(rx_signal);
% % % %     Rx_signal_fft = fft(rx_signal);
% % % %     f_rx = (-N_fft/2:N_fft/2-1)*(fs/N_fft);
% % % % 
% % % %     subplot(2, 1, 2);
% % % %     plot(f_rx, fftshift(abs(Rx_signal_fft)), 'r', 'LineWidth', 1.5);
% % % %     title('匹配滤波后的信号 rx\_signal 的双边频谱图');
% % % %     xlabel('频率 (Hz)');
% % % %     ylabel('幅度');
% % % %     grid on;
% % % % 
% % % %     %% 下采样
% % % %     downsampled_rx_signal = rx_signal(1:sps2:end); % 对信号进行下采样
% % % % 
% % % %     %% 绘制downsampled_rx_signal的波形和频谱图
% % % %     figure;
% % % %     subplot(2, 1, 1);
% % % %     plot(1:length(downsampled_rx_signal), downsampled_rx_signal, 'b', 'LineWidth', 1.5);
% % % %     title('下采样后的信号 downsampled\_rx\_signal');
% % % %     xlabel('样本点');
% % % %     ylabel('幅度');
% % % %     grid on;
% % % % 
% % % %     N_fft = length(downsampled_rx_signal);
% % % %     Downsampled_rx_signal_fft = fft(downsampled_rx_signal);
% % % %     f_downsampled_rx = (-N_fft/2:N_fft/2-1)*(fs/N_fft);
% % % % 
% % % %     subplot(2, 1, 2);
% % % %     plot(f_downsampled_rx, fftshift(abs(Downsampled_rx_signal_fft)), 'r', 'LineWidth', 1.5);
% % % %     title('下采样后的信号 downsampled\_rx\_signal 的双边频谱图');
% % % %     xlabel('频率 (Hz)');
% % % %     ylabel('幅度');
% % % %     grid on;
% % % % 
% % % %     %% BPSK解调
% % % %     rx_data = BPSK_Demodulate(downsampled_rx_signal); % 进行BPSK解调
% % % % 
% % % %     %% 计算误比特率
% % % %     x_bin = x_bin(1:length(rx_data)); % 截断原始数据长度以匹配接收数据
% % % %     ber(i) = sum(abs(x_bin - rx_data)) / length(x_bin); % 计算误比特率
% % % % 
% % % %     %% 打印当前信噪比下的误比特率
% % % %     fprintf('SNR = %d dB, BER = %e\n', SNR_dB(i), ber(i));
% % % % end
% % % % 
% % % % %% 绘制误比特率曲线
% % % % figure;
% % % % plot(SNR_dB, ber, 'o-');
% % % % xlabel('SNR (dB)');
% % % % ylabel('BER');
% % % % title('BER vs. SNR');
% % % % grid on;
% % % % 
% % % % %% 绘制原始信源数据和解调后的数据的对比图
% % % % figure;
% % % % subplot(2, 1, 1);
% % % % plot(1:length(x_bin), x_bin, 'b', 'LineWidth', 1.5);
% % % % title('原始信源数据');
% % % % xlabel('样本点');
% % % % ylim([-0.5 1.5]);
% % % % 
% % % % subplot(2, 1, 2);
% % % % plot(1:length(rx_data), rx_data, 'r', 'LineWidth', 1.5);
% % % % title('解调后的数据');
% % % % xlabel('样本点');
% % % % ylabel('二进制数据');
% % % % ylim([-0.5 1.5]);
% % 
% % 
% % 
% % clc;
% % clear all;
% % 
% % %% 初始化参数设置
% % rolloff_factor = 0.5;    % 滚降因子
% % fs = 48000;              % 采样频率
% % fc = 1800;               % 载波频率
% % T_s = 1/fs;              % 信道采样间隔
% % condition = 2;           % 信道条件
% % N = 10000;                % 符号数
% % sps1 = 8;                % 第一次上采样倍数
% % sps2 = 20;               % 第二次上采样后的总倍数
% % phase_bits = 16;         % 相位累加器位数
% % 
% % %% 生成信源数据
% % x_bin = randi([0, 1], 1, N); % 生成随机二进制序列
% % 
% % %% BPSK调制
% % x_t = BPSK_Modulate(x_bin); % 将二进制数据进行BPSK调制
% % 
% % %% 第一次上采样（8倍）
% % x_t_upsampled1 = upsample(x_t, sps1); % 进行第一次上采样
% % 
% % %% 成型滤波
% % rcos_fir = rcosdesign(rolloff_factor, 8, sps1, 'sqrt'); % 生成根升余弦滤波器
% % shaped_bpsk_message1 = conv(x_t_upsampled1, rcos_fir, 'same'); % 进行成型滤波
% % 
% % %% 第二次上采样（在8倍上采样的基础上进行2.5倍上采样）
% % shaped_bpsk_message2 = resample(shaped_bpsk_message1, sps2, sps1); % 进行第二次上采样
% % 
% % %% 相位累加器
% % phase_increment = 2 * pi * fc / fs; % 相位增量
% % phase_accumulator = 0; % 初始化相位累加器
% % phase_values = zeros(1, length(shaped_bpsk_message2));
% % 
% % for n = 1:length(phase_values)
% %     % 更新相位累加器
% %     phase_accumulator = mod(phase_accumulator + phase_increment, 2*pi);
% %     % 存储当前相位值
% %     phase_values(n) = phase_accumulator;
% % end
% % 
% % %% 生成载波信号
% % carrier = cos(phase_values);
% % 
% % %% 将信号乘上载波
% % tx_signal = shaped_bpsk_message2 .* carrier; % 调制信号乘以载波
% % 
% % %% NBHF信道传输
% % y_nbhf = NBHF_channel(tx_signal, T_s, condition); % 通过NBHF信道传输
% % 
% % %% 设置不同信噪比范围
% % SNR_dB = 10:1:30; % 设置信噪比范围和步长
% % ber = zeros(1, length(SNR_dB)); % 初始化误比特率数组
% % 
% % for i = 1:length(SNR_dB)
% %     %% 添加高斯白噪声
% %     y_filtered = awgn(y_nbhf, SNR_dB(i), 'measured'); % 添加高斯白噪声
% % 
% %     %% 下变频
% %     y_filtered = y_filtered .* carrier; % 将接收信号乘以载波进行下变频
% % 
% %     %% 匹配滤波
% %     rx_signal = conv(y_filtered, rcos_fir, 'same'); % 进行匹配滤波
% % 
% %     %% 下采样
% %     downsampled_rx_signal = rx_signal(1:sps2:end); % 对信号进行下采样
% % 
% %     %% BPSK解调
% %     rx_data = BPSK_Demodulate(downsampled_rx_signal); % 进行BPSK解调
% % 
% %     %% 计算误比特率
% %     x_bin = x_bin(1:length(rx_data)); % 截断原始数据长度以匹配接收数据
% %     ber(i) = sum(abs(x_bin - rx_data)) / length(x_bin); % 计算误比特率
% % 
% %     %% 打印当前信噪比下的误比特率
% %     fprintf('SNR = %d dB, BER = %e\n', SNR_dB(i), ber(i));
% % end
% % 
% % %% 绘制误比特率曲线
% % figure;
% % plot(SNR_dB, ber, 'o-');
% % xlabel('SNR (dB)');
% % ylabel('BER');
% % title('BER vs. SNR');
% % grid on;
% % 
% % %% 绘制原始信源数据和解调后的数据的对比图
% % figure;
% % subplot(2, 1, 1);
% % plot(1:length(x_bin), x_bin, 'b', 'LineWidth', 1.5);
% % title('原始信源数据');
% % xlabel('样本点');
% % ylim([-0.5 1.5]);
% % 
% % subplot(2, 1, 2);
% % plot(1:length(rx_data), rx_data, 'r', 'LineWidth', 1.5);
% % title('解调后的数据');
% % xlabel('样本点');
% % ylabel('二进制数据');
% % ylim([-0.5 1.5]');
% 
% clc;
% clear all;
% 
% % 参数定义
% fs = 48000;         % 采样频率
% fc = 1800;          % 载波频率
% N = 2400;           % 符号数
% sps1 = 8;           % 第一次上采样倍数
% sps2 = 20;          % 第二次上采样后的总倍数
% rolloff_factor = 0.5; % 滚降因子
% carrier_length = 20; % 载波信号的采样点数
% 
% % 产生随机二进制01信息
% x_bin = randi([0 1], 1, N); % 生成1*N的行向量
% 
% % BPSK调制
% x_t = BPSK_Modulate(x_bin);
% 
% % 第一次上采样（8倍）
% x_t_upsampled1 = upsample(x_t, sps1);
% 
% % 成型滤波
% rcos_fir = rcosdesign(rolloff_factor, 8, sps1, 'sqrt'); % 根升余弦滤波器
% shaped_bpsk_message1 = conv(x_t_upsampled1, rcos_fir, 'same'); % 进行成型滤波
% 
% % 第二次上采样（在8倍上采样的基础上进行2.5倍上采样）
% shaped_bpsk_message2 = resample(shaped_bpsk_message1, sps2, sps1);
% 
% % 计算相位增量
% phase_increment = 2 * pi * fc * 20 / fs;
% 
% % 初始化相位累加器
% phase_accumulator = 0;
% carrier = zeros(1, length(shaped_bpsk_message2));  % 初始化载波信号
% 
% % 生成载波信号并与基带信号点乘
% tx_signal = zeros(1, length(shaped_bpsk_message2));
% for i = 1:length(shaped_bpsk_message2)
%     carrier(i) = cos(phase_accumulator);
%     tx_signal(i) = shaped_bpsk_message2(i) * carrier(i);
%     phase_accumulator = phase_accumulator + phase_increment;
%     if phase_accumulator >= 2 * pi
%         phase_accumulator = phase_accumulator - 2 * pi;  % 相位累加器溢出处理
%     end
% end
% 
% % 绘制调制后的信号的前100个样本点
% figure;
% plot(tx_signal(1:100)); % 仅绘制前100个样本点
% title('Transmitted Signal with Continuous Phase Carrier');
% xlabel('Sample Number');
% ylabel('Amplitude');
% grid on;
% 
% % 绘制载波信号的前100个样本点
% figure;
% plot(carrier(1:100)); % 仅绘制前100个样本点
% title('Carrier Signal');
% xlabel('Sample Number');
% ylabel('Amplitude');
% grid on;
% 
% % 绘制相位累加器的前200个值
% num_samples_to_plot = 200;
% phase_values = zeros(1, num_samples_to_plot);
% phase_accumulator = 0;
% for i = 1:num_samples_to_plot
%     phase_values(i) = phase_accumulator;
%     phase_accumulator = phase_accumulator + phase_increment;
%     if phase_accumulator >= 2 * pi
%         phase_accumulator = phase_accumulator - 2 * pi;
%     end
% end
% 
% figure;
% plot(1:num_samples_to_plot, phase_values);
% xlabel('Sample Number');
% ylabel('Phase (radians)');
% title('Phase Accumulator Values (First 200 Samples)');
% grid on;


% clc;
% clear all;
% 
% %% 初始化参数设置
% rolloff_factor = 0.5;    % 滚降因子
% fs = 48000;              % 采样频率
% fc = 1800;               % 载波频率
% T_s = 1/fs;              % 信道采样间隔
% condition = 1;           % 信道条件
% N = 2400;                % 符号数
% SNR_dB = 30;             % 信噪比
% sps1 = 8;                % 第一次上采样倍数
% sps2 = 20;               % 第二次上采样后的总倍数
% phase_bits = 16;         % 相位累加器位数
% 
% % 产生随机二进制01信息
% x_bin = randi([0 1], 1, N); % 生成1*N的行向量
% 
% % BPSK调制
% x_t = BPSK_Modulate(x_bin);
% 
% % 第一次上采样（8倍）
% x_t_upsampled1 = upsample(x_t, sps1);
% 
% % 成型滤波
% rcos_fir = rcosdesign(rolloff_factor, 8, sps1, 'sqrt'); % 根升余弦滤波器
% shaped_bpsk_message1 = conv(x_t_upsampled1, rcos_fir, 'same'); % 进行成型滤波
% 
% % 第二次上采样（在8倍上采样的基础上进行2.5倍上采样）
% shaped_bpsk_message2 = resample(shaped_bpsk_message1, sps2, sps1);
% 
% %% 计算相位增量和相位跳变
% phase_step = 2 * pi * fc / fs; % 相位步长
% phase_jump = 2 * pi - (20 * phase_step); % 相位跳变
% 
% %% 生成相位累加器的载波信号并进行点乘
% N_samples = length(shaped_bpsk_message2);
% phase_accumulator = 0;  % 初始化相位累加器
% 
% tx_signal = zeros(1, N_samples); % 初始化调制信号
% phase_values = zeros(1, N_samples); % 用于存储相位值以便绘图
% for i = 1:N_samples
%     carrier = cos(phase_accumulator); % 生成载波信号
%     tx_signal(i) = shaped_bpsk_message2(i) * carrier; % 将信号逐点乘上载波
%     phase_values(i) = phase_accumulator; % 存储当前相位值
%     phase_accumulator = phase_accumulator + phase_step; % 更新相位累加器
%     if mod(i, 20) == 0
%         phase_accumulator = phase_accumulator + phase_jump; % 修正相位跳变
%     end
% end
% 
% 
% 
% % 绘制调制后的信号的前100个样本点
% figure;
% plot(tx_signal(1:100)); % 仅绘制前100个样本点
% title('Transmitted Signal with Continuous Phase Carrier');
% xlabel('Sample Number');
% ylabel('Amplitude');
% grid on;
% 
% % 绘制载波信号的前100个样本点
% figure;
% plot(carrier(1:100)); % 仅绘制前100个样本点
% title('Carrier Signal');
% xlabel('Sample Number');
% ylabel('Amplitude');
% grid on;
% 
% % 绘制相位累加器的前200个值
% num_samples_to_plot = 200;
% phase_values = zeros(1, num_samples_to_plot);
% phase_accumulator = 0;
% for i = 1:num_samples_to_plot
%     phase_values(i) = phase_accumulator;
%     phase_accumulator = mod(phase_accumulator + phase_increment, 2 * pi);
% end
% 
% figure;
% plot(1:num_samples_to_plot, phase_values);
% xlabel('Sample Number');
% ylabel('Phase (radians)');
% title('Phase Accumulator Values (First 200 Samples)');
% grid on;
% 
% %% 通过信道传输
% % 确保 NBHF_channel 函数已正确实现
% % y_nbhf = NBHF_channel(tx_signal, T_s, condition); % 使用定义的信道函数
% 
% % % 添加高斯白噪声
% % y_noise = awgn(y_nbhf, SNR_dB, 'measured'); % 添加高斯白噪声


% clc;
% clear all;
% 
% %% 初始化参数设置
% rolloff_factor = 0.5;    % 滚降因子
% fs = 48000;              % 采样频率
% fc = 1800;               % 载波频率
% T_s = 1/fs;              % 信道采样间隔
% condition = 2;           % 信道条件
% N = 10000;               % 符号数
% sps1 = 8;                % 第一次上采样倍数
% sps2 = 20;               % 第二次上采样后的总倍数
% 
% %% 生成信源数据
% x_bin = randi([0, 1], 1, N); % 生成随机二进制序列
% 
% %% BPSK调制
% x_t = BPSK_Modulate(x_bin); % 将二进制数据进行BPSK调制
% 
% %% 第一次上采样（8倍）
% x_t_upsampled1 = upsample(x_t, sps1); % 进行第一次上采样
% 
% %% 成型滤波
% rcos_fir = rcosdesign(rolloff_factor, 8, sps1, 'sqrt'); % 生成根升余弦滤波器
% shaped_bpsk_message1 = conv(x_t_upsampled1, rcos_fir, 'same'); % 进行成型滤波
% 
% %% 第二次上采样（在8倍上采样的基础上进行2.5倍上采样）
% shaped_bpsk_message2 = resample(shaped_bpsk_message1, sps2, sps1); % 进行第二次上采样
% 
% %% 计算相位增量和相位跳变
% phase_step = 2 * pi * fc / fs; % 相位步长
% phase_jump = 2 * pi - (20 * phase_step); % 相位跳变
% 
% %% 生成相位累加器的载波信号并进行点乘
% N_samples = length(shaped_bpsk_message2);
% phase_accumulator = 0;  % 初始化相位累加器
% 
% tx_signal = zeros(1, N_samples); % 初始化调制信号
% phase_values = zeros(1, N_samples); % 用于存储相位值以便绘图
% for i = 1:N_samples
%     carrier = cos(phase_accumulator); % 生成载波信号
%     tx_signal(i) = shaped_bpsk_message2(i) * carrier; % 将信号逐点乘上载波
%     phase_values(i) = phase_accumulator; % 存储当前相位值
%     phase_accumulator = phase_accumulator + phase_step; % 更新相位累加器
%     if mod(i, 20) == 0
%         phase_accumulator = phase_accumulator + phase_jump; % 修正相位跳变
%     end
% end
% 
% %% 显示相位跳变值
% disp(['相邻两点的相位增量: ', num2str(phase_step)]);
% disp(['相位跳变值 q: ', num2str(phase_jump)]);
% 
% %% 绘制载波相位图（截取前100个采样点）
% figure;
% plot(0:99, phase_values(1:100));
% title('载波相位随时间变化图（前100个采样点）');
% xlabel('采样点');
% ylabel('相位 (弧度)');
% grid on;
% 
% %% 绘制载波信号波形图（截取前1000个采样点）
% figure;
% plot(0:999, tx_signal(1:1000));
% title('载波信号波形图（前1000个采样点）');
% xlabel('采样点');
% ylabel('幅度');
% grid on;
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 












clc;
clear;

%% 初始化参数设置
rolloff_factor = 0.3;    % 调整滚降因子
fs = 48000;              % 采样频率
fc = 1800;            % 载波频率
T_s = 1/fs;              % 信道采样间隔
condition = 2;           % 信道条件
N = 1200;                % 符号数
sps1 = 40;               % 第一次上采样倍数

%% 生成载波
t = 0:1/fs:10;
CarryFreq = exp(1i*2*pi*fc*t);

%% 设置信噪比范围
SNR_dB = -10:1:10; % 设置信噪比范围和步长
ber = zeros(1, length(SNR_dB)); % 初始化误比特率数组
total_time = 20; % 总时间为30秒

%% 30秒的循环
for second = 1:total_time
    %% 产生随机二进制01信息
    x_bin = randi([0 1], 1, N); 

    %% 调制
    x_t = BPSK_Modulate(x_bin);

    %% 第一次上采样（40倍）
    x_t_upsampled1 = upsample(x_t, sps1);

    %% 成型滤波
    rcos_fir = rcosdesign(rolloff_factor, 8, sps1, 'sqrt');
    shaped_qam4_message1 = conv(x_t_upsampled1, rcos_fir, 'same');

    %% 乘载波
    tx_signal = shaped_qam4_message1 .* CarryFreq(1:length(shaped_qam4_message1));

    %% 通过信道传输
    y_nbhf = NBHF_channel(tx_signal, T_s, condition); % 使用定义的信道函数

    for i = 1:length(SNR_dB)
        %% 添加高斯白噪声
        y_noise = awgn(y_nbhf, SNR_dB(i), 'measured'); % 添加高斯白噪声

        %% 下变频
        y_noise = y_noise .* conj(CarryFreq(1:length(shaped_qam4_message1)));

        %% 匹配滤波
        rx_signal = conv(y_noise, rcos_fir, 'same');

        %% 下采样
        downsampled_rx_signal = rx_signal(1:sps1:end); % 对信号进行下采样

        %% 解调
        rx_data = BPSK_Demodulate(downsampled_rx_signal); 

        %% 计算误比特率并累积
        x_bin = x_bin(1:length(rx_data)); % 截断原始数据长度以匹配接收数据

        ber(i) = ber(i) + sum(abs(x_bin - rx_data)) / length(x_bin); % 累积误比特率

        %% 打印当前秒数和信噪比下的误比特率
        fprintf('Second = %d, SNR = %d dB, BER = %e\n', second, SNR_dB(i), ber(i)/second);
    end
end

%% 计算平均误比特率
ber = ber / total_time;

% save("chuantongxitong2.mat","SNR_dB","ber");

%% 绘制平均误比特率曲线
figure;
plot(SNR_dB, ber, 'o-');
xlabel('SNR (dB)');
ylabel('平均BER');
title('平均BER vs. SNR');
grid on;