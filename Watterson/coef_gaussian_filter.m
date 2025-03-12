function h = coef_gaussian_filter(F_2_sigma, Ts, order)
% Function: coef_gaussian_filter
% 高斯滤波器系数计算函数
%
% Input:
%   F_2_sigma: 两倍标准差频率
%   Ts: 采样时间间隔
%   order: 滤波器阶数（长度）
%
% Output:
%   h: 高斯滤波器的脉冲响应系数

Fs = 1 / Ts;  % 采样频率

% N = ceil(K_qual * Fs / F_2_sigma) + 1; % 原始计算方式，用于确定滤波器长度
N = order;  % 直接使用提供的阶数

N_half = (N - 1) / 2;  % 滤波器长度的一半
n = 0:N-1;  % 滤波器系数的离散时间索引

sigma = Fs * sqrt(2) / (2 * pi * F_2_sigma);  % 高斯函数的标准差

% 高斯滤波器的脉冲响应
mult = N_half;
h = (dnorm(n, mult, sigma) / dnorm(0, 0, sigma)) / (sqrt(2 * pi) * sigma);

K_enb = 1 / (4 * dnorm(0, 0, 1));  % 1Hz 两倍标准差滤波器的等效噪声带宽
Gain = sqrt(Fs / (2 * F_2_sigma * K_enb));  % 增益因子，用于保持滤波器的单位增益
h = h * Gain;  % 缩放滤波器系数，以达到期望的增益
end
