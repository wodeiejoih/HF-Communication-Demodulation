function dout = FIR_filter(h,datin)

n = length(datin);
h_len = length(h);
temp = zeros(1,h_len);

for k = 1:n
    temp(2:end) = temp(1:end-1);
    temp(1) = datin(k);
    dout(k) = sum(temp.*h);
end