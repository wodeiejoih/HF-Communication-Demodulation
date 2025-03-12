function y_t = NBHF_channel(x_t_noisy, T_s, condition)

% 信道采样间隔:
Fs = 1/T_s;
% 时间偏移
t_0 = 0;
% seed为线型同余法产生均匀分布随机数的种子
seed = 1;
% Shape Quality Constant KQUAL=1.4 (larger the constant, the better the match)
K_qual = 1.5;  
% 根据选择的信道类型确定信道特性参数，共10种信道环境
[Doppler_spread, Doppler_shift, Delay] = channel_para(condition);
% 将延时转换为延时位数
delay = round(Delay/T_s);
% FIR Length
N = ceil(K_qual*Fs/(Doppler_spread*2))+1; 
% 计算输出数据长度
DATA = length(x_t_noisy);
t = (0:DATA-1)*T_s;

% 计算Gaussian滤波器系数
h = coef_gaussian_filter(Doppler_spread*2, T_s, N);

rand_init(seed);    % 为线型同余法产生随机数进行初始化，可根据程序的不同改变其位置
for k = 1:2
    % 产生高斯分布白噪声序列
    rand_uni1 = rand1(1, DATA);
    rand_uni2 = rand2(1, DATA);
    % Box-Muller变换
    rand_g1 = sqrt(-2*log(rand_uni1)).*cos(2*pi*rand_uni2);
    rand_g2 = sqrt(-2*log(rand_uni1)).*sin(2*pi*rand_uni2);
    rand_g = rand_g1 + 1i*rand_g2;
    rand_g = rand_g*sqrt(0.5);
    randnums(:, k) = FIR_filter(h, rand_g).*exp(1i*2*pi*Doppler_shift*t);%randnums 矩阵存储了生成的高斯噪声序列，并经过 FIR 滤波和多普勒频移处理，模拟了信道的多径效应和多普勒效应
end
ht = sqrt(0.5)*randnums;%信道脉冲响应 ht：信道脉冲响应表示信号在各个路径上的衰减和延迟。

% 产生两路信号，一路为输入信号，一路为输入信号的延时
x_t1 = x_t_noisy;%代表信号通过直接路径到达接收机。
x_t2 = [zeros(1, delay), x_t_noisy(delay+1:end)];%
x_t2 = x_t2(1:DATA); % 确保 x_t2 的长度与 x_t1 一致

% 计算输出信号。在无线通信中，信号通过多个路径到达接收端，每条路径具有不同的延迟和衰减。接收到的信号是这些路径信号的叠加，这种现象称为多径效应
y1 = x_t1.*ht(:, 1)';%直接路径信号 x_t1 乘以信道脉冲响应的第一个路径 ht(:, 1)，模拟信号在直接路径上的衰减和相位变化
% y2 = x_t2.*ht(:, 2)';%延迟路径信号乘以信道脉冲响应的第二个路径 ht(:, 2)，模拟信号在延迟路径上的衰减和相位变化。
y_t = y1 ;
end



