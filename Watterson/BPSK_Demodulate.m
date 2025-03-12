function [BPSK_Demodulate_R]=BPSK_Demodulate(y_t_noisy)
%˵��: �˺������ڽ�signal����BPSK���
%BPSK_Modulate_R: Ҫ����BPSK����������ź�
k=length(y_t_noisy);  %��ȡ����
BPSK_Demodulate_R=zeros(1,k);
real_part=zeros(1,k);
real_part=real(y_t_noisy);   %��ȡʵ��
real_part(1,real_part>=0)=1;    %ʵ���У����ڵ���0��Ԫ������Ϊ1
real_part(1,real_part<0)=-1;      %ʵ���У�С��0��Ԫ������Ϊ-1

%����t=0ʱ����λΪ�У�������ʱԼ���õ�
if real_part(1,1)==-1 %��ǰ��λ��ǰһʱ����λ��ͬ
    BPSK_Demodulate_R(1,1)=0;   %���ֵΪ0
else
    BPSK_Demodulate_R(1,1)=1;   %��ͬ����ֵΪ1
end

for n=2:k
    if real_part(1,n)==real_part(1,n-1) %��ǰ��λ��ǰһʱ����λ��ͬ
        BPSK_Demodulate_R(1,n)=0;   %���ֵΪ0
    else
        BPSK_Demodulate_R(1,n)=1;   %��ͬ����ֵΪ1
    end
end



