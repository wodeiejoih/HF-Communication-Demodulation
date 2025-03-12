function y_t = NBHF_channel(x_t_noisy, T_s, condition)

% �ŵ��������:
Fs = 1/T_s;
% ʱ��ƫ��
t_0 = 0;
% seedΪ����ͬ�෨�������ȷֲ������������
seed = 1;
% Shape Quality Constant KQUAL=1.4 (larger the constant, the better the match)
K_qual = 1.5;  
% ����ѡ����ŵ�����ȷ���ŵ����Բ�������10���ŵ�����
[Doppler_spread, Doppler_shift, Delay] = channel_para(condition);
% ����ʱת��Ϊ��ʱλ��
delay = round(Delay/T_s);
% FIR Length
N = ceil(K_qual*Fs/(Doppler_spread*2))+1; 
% ����������ݳ���
DATA = length(x_t_noisy);
t = (0:DATA-1)*T_s;

% ����Gaussian�˲���ϵ��
h = coef_gaussian_filter(Doppler_spread*2, T_s, N);

rand_init(seed);    % Ϊ����ͬ�෨������������г�ʼ�����ɸ��ݳ���Ĳ�ͬ�ı���λ��
for k = 1:2
    % ������˹�ֲ�����������
    rand_uni1 = rand1(1, DATA);
    rand_uni2 = rand2(1, DATA);
    % Box-Muller�任
    rand_g1 = sqrt(-2*log(rand_uni1)).*cos(2*pi*rand_uni2);
    rand_g2 = sqrt(-2*log(rand_uni1)).*sin(2*pi*rand_uni2);
    rand_g = rand_g1 + 1i*rand_g2;
    rand_g = rand_g*sqrt(0.5);
    randnums(:, k) = FIR_filter(h, rand_g).*exp(1i*2*pi*Doppler_shift*t);%randnums ����洢�����ɵĸ�˹�������У������� FIR �˲��Ͷ�����Ƶ�ƴ���ģ�����ŵ��ĶྶЧӦ�Ͷ�����ЧӦ
end
ht = sqrt(0.5)*randnums;%�ŵ�������Ӧ ht���ŵ�������Ӧ��ʾ�ź��ڸ���·���ϵ�˥�����ӳ١�

% ������·�źţ�һ·Ϊ�����źţ�һ·Ϊ�����źŵ���ʱ
x_t1 = x_t_noisy;%�����ź�ͨ��ֱ��·��������ջ���
x_t2 = [zeros(1, delay), x_t_noisy(delay+1:end)];%
x_t2 = x_t2(1:DATA); % ȷ�� x_t2 �ĳ����� x_t1 һ��

% ��������źš�������ͨ���У��ź�ͨ�����·��������նˣ�ÿ��·�����в�ͬ���ӳٺ�˥�������յ����ź�����Щ·���źŵĵ��ӣ����������Ϊ�ྶЧӦ
y1 = x_t1.*ht(:, 1)';%ֱ��·���ź� x_t1 �����ŵ�������Ӧ�ĵ�һ��·�� ht(:, 1)��ģ���ź���ֱ��·���ϵ�˥������λ�仯
% y2 = x_t2.*ht(:, 2)';%�ӳ�·���źų����ŵ�������Ӧ�ĵڶ���·�� ht(:, 2)��ģ���ź����ӳ�·���ϵ�˥������λ�仯��
y_t = y1 ;
end



