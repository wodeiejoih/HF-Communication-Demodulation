function [BPSK_Modulate_R]=BPSK_Modulate(bit_sequence)
%˵��: �˺������ڽ�signal����BPSK����
%signal: Ҫ����BPSK���Ƶ������ź�
k=length(bit_sequence);
BPSK_Modulate_R=zeros(1,k);  %������BPSK��ʽ���ƺ�Ĳ������ݵ�����
%���ó�ʼ�Ƚ���λ������t=0ʱ����λΪ��
if bit_sequence(1,1)==0  
    %�ź�Ϊ0�������ֵ(��λ)��ǰһʱ����ͬ
    BPSK_Modulate_R(1,1)=-1;  
else
    %�ź�Ϊ1�������ֵ(��λ)��ǰһʱ������
    BPSK_Modulate_R(1,1)=1;
end
for n=2:k
    if bit_sequence(1,n)==0  
        %�ź�Ϊ0�������ֵ(��λ)��ǰһʱ����ͬ
        BPSK_Modulate_R(1,n)=BPSK_Modulate_R(1,n-1);  
    else
        %�ź�Ϊ1�������ֵ(��λ)��ǰһʱ������
        BPSK_Modulate_R(1,n)=-BPSK_Modulate_R(1,n-1);
    end
end