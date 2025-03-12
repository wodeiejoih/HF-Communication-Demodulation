function [BPSK_Modulate_R]=BPSK_Modulate(bit_sequence)
%说明: 此函数用于将signal进行BPSK调制
%signal: 要进行BPSK调制的输入信号
k=length(bit_sequence);
BPSK_Modulate_R=zeros(1,k);  %放置以BPSK方式调制后的波形数据的数组
%设置初始比较相位，假设t=0时刻相位为π
if bit_sequence(1,1)==0  
    %信号为0，则调制值(相位)与前一时刻相同
    BPSK_Modulate_R(1,1)=-1;  
else
    %信号为1，则调制值(相位)与前一时刻相差π
    BPSK_Modulate_R(1,1)=1;
end
for n=2:k
    if bit_sequence(1,n)==0  
        %信号为0，则调制值(相位)与前一时刻相同
        BPSK_Modulate_R(1,n)=BPSK_Modulate_R(1,n-1);  
    else
        %信号为1，则调制值(相位)与前一时刻相差π
        BPSK_Modulate_R(1,n)=-BPSK_Modulate_R(1,n-1);
    end
end