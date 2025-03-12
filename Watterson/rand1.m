function randnum = rand1(M,N)

global seed1 seed2 seed3
if(nargin == 0)
    row = 1;
    column = 1;
elseif(nargin == 1)
    row = 1;
    column = M;
else
    row = M;
    column = N;
end

for m = 1:row
    for n = 1:column
        seed1 = mod(171*seed1,30269);
        seed2 = mod(172*seed2,30307);
        seed3 = mod(170*seed3,30323);
        randnum(m,n) = mod(seed1/30269+seed2/30307+seed3/30323,1);
    end
end