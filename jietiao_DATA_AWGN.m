clc;
clear all;

%% 初始化参数设置
rolloff_factor = 0.3;    % 调整滚降因子
fs = 48000;              % 采样频率
fc = 1800;
T_s = 1/fs;              % 信道采样间隔
condition = 2;           % 信道条件
N = 1200;                % 符号数（每秒发送的比特数）
sps1 = 40;               % 每个符号的采样点数
total_time = 1200;        % 总时间（秒）

%% 预分配空间，定义所有变量
all_SNR_dB_range = 0;  % 信噪比范围从 -10 到 10

for SNR_dB = all_SNR_dB_range

    % 预分配用于保存所有数据的空间
    all_snr_training_data_real = zeros(sps1, N * total_time);
    all_snr_training_data_imag = zeros(sps1, N * total_time);
    all_labels = zeros(1, N * total_time);
    all_SNR_dB = SNR_dB * ones(1, total_time);  % 保存当前 SNR
    all_x_bin = zeros(total_time, N);  % 新增的数组，用于保存每秒的x_bin

    % 生成载波信号
    t = 0:1/fs:(N * sps1 - 1)/fs;
    CarryFreq = exp(1i * 2 * pi * fc * t);

    %% 1200秒的循环
    for second = 1:total_time
        %% 产生随机二进制01信息
        x_bin = randi([0 1], 1, N); % 生成长度为N的二进制序列
        all_x_bin(second, :) = x_bin; % 保存x_bin到数组中

        %% 调制
        x_t = BPSK_Modulate(x_bin); % BPSK调制

        %% 第一次上采样（40倍）
        x_t_upsampled1 = upsample(x_t, sps1); % 将调制信号上采样40倍

        %% 成型滤波
        rcos_fir = rcosdesign(rolloff_factor, 9, sps1, 'sqrt'); % 根升余弦滤波器
        shaped_bpsk_message1 = conv(x_t_upsampled1, rcos_fir, 'same'); % 进行卷积成型滤波

        %% 乘载波
        tx_signal = shaped_bpsk_message1 .* CarryFreq(1:length(shaped_bpsk_message1)); % 乘以载波信号

        %% 通过信道传输
        % y_nbhf = NBHF_channel(tx_signal, T_s, condition); % 通过信道传输

        %% 添加高斯白噪声
        y_noise = awgn(tx_signal, SNR_dB, 'measured'); % 添加高斯白噪声

        %% 下变频
        y_noise = y_noise .* conj(CarryFreq(1:length(shaped_bpsk_message1))); % 与载波共轭相乘，完成下变频

        %% 匹配滤波
        rx_signal = conv(y_noise, rcos_fir, 'same'); % 进行匹配滤波




        %% 重塑
        snr_training_data_real = reshape(real(rx_signal), [sps1, N]); % 将接收到的信号重塑为sps1行，N列（实部）
        snr_training_data_imag = reshape(imag(rx_signal), [sps1, N]); % 将接收到的信号重塑为sps1行，N列（虚部）

        %% 标签
        x_t = x_t';

        %% 确保标签是两个类别
        x_t = ensure_two_classes(x_t);

        %% 将数据填充到预分配的数组中
        start_idx = (second - 1) * N + 1;
        end_idx = second * N;
        all_snr_training_data_real(:, start_idx:end_idx) = snr_training_data_real;
        all_snr_training_data_imag(:, start_idx:end_idx) = snr_training_data_imag;
        all_labels(start_idx:end_idx) = x_t;

        % 打印当前秒数
        fprintf('信噪比 %d dB: 第 %d 秒数据已生成并保存\n', SNR_dB, second);
    end

    %% 归一化操作
    [all_snr_training_data_real, real_ps] = mapminmax(all_snr_training_data_real);
    [all_snr_training_data_imag, imag_ps] = mapminmax(all_snr_training_data_imag);

    %% 打印归一化参数
    fprintf('信噪比 %d dB: 实部数据的归一化范围：[%f, %f]\n', SNR_dB, real_ps.ymin, real_ps.ymax);
    fprintf('信噪比 %d dB: 虚部数据的归一化范围：[%f, %f]\n', SNR_dB, imag_ps.ymin, imag_ps.ymax);

    %% 将 all_labels 转换为分类标签
    all_labels = categorical(all_labels);

    %% 保存数据
    save_filename = sprintf('jietiao_data_xunlian_%ddB_1200s_.mat', SNR_dB);
    save(save_filename, 'all_snr_training_data_real', 'all_snr_training_data_imag', 'all_labels', 'all_SNR_dB', 'all_x_bin', 'sps1', 'N', 'total_time', 'real_ps', 'imag_ps');
    fprintf('信噪比 %d dB 数据保存成功：%s\n', SNR_dB, save_filename);

end

%% 确保标签是两个类别
function [x_t] = ensure_two_classes(x_t)
    unique_classes = unique(x_t);
    if numel(unique_classes) ~= 2
        error('标签不是两个类别');
    end
end
