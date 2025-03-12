function [BPSK_Demodulate_R]=BPSK_Demodulate(y_t_noisy)
%说明: 此函数用于将signal进行BPSK解调
%BPSK_Modulate_R: 要进行BPSK解调的输入信号
k=length(y_t_noisy);  %获取长度
BPSK_Demodulate_R=zeros(1,k);
real_part=zeros(1,k);
real_part=real(y_t_noisy);   %获取实部
real_part(1,real_part>=0)=1;    %实部中，大于等于0的元素则置为1
real_part(1,real_part<0)=-1;      %实部中，小于0的元素则置为-1

%假设t=0时刻相位为π，跟调制时约定好的
if real_part(1,1)==-1 %当前相位与前一时刻相位相同
    BPSK_Demodulate_R(1,1)=0;   %解调值为0
else
    BPSK_Demodulate_R(1,1)=1;   %不同则解调值为1
end

for n=2:k
    if real_part(1,n)==real_part(1,n-1) %当前相位与前一时刻相位相同
        BPSK_Demodulate_R(1,n)=0;   %解调值为0
    else
        BPSK_Demodulate_R(1,n)=1;   %不同则解调值为1
    end
end



