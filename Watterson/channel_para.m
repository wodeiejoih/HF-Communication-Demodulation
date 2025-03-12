function [spread,shift,delay] = channel_para(condition)

switch condition
    case 1
        spread = 0.5 ;shift = 0;delay = 0.5 ;
    case 2
        spread = 1.5;shift = 0;delay = 2;
    case 3
        spread = 10;shift = 0;delay = 6;
    case 4
        spread = 0.1;shift = 0;delay = 0.5;
    case 5
        spread = 0.5;shift = 0;delay = 1;
    case 6
        spread = 1;shift = 0;delay = 2;
    case 7
        spread = 1;shift = 0;delay = 7;
    case 8
        spread = 0.5;shift = 0;delay = 1;
    case 9
        spread = 10;shift = 0;delay = 3;
    case 10
        spread = 30;shift = 0;delay = 7;
    otherwise
        disp('Please input a correct channel condition!')
end