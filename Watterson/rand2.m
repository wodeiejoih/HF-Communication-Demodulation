function randnum = rand2(M,N)

global seed4 seed5
if(nargin == 0)
    row = 1;
    column = 1;
elseif(nargin == 1)
    row = 1;
    colunm = M;
else
    row = M;
    column = N;
end

for m = 1:row
    for n = 1:column
        k = fix(seed4/53668);
        seed4 = 40014*(seed4-k*53668)-k*12211;
        if(seed4 < 0)
            seed4 = seed4+2147483563;
        end
        k = fix(seed5/52774);
        seed5 = 40692*(seed5-k*52774)-k*3791;
        if(seed5 < 0)
            seed5 = seed5+2147473399;
        end        
        w = seed5-seed4;
        if(w <= 0)
            w = w+2147483562;
        end
        randnum(m,n) = w*4.656613057392e-10;
    end
end