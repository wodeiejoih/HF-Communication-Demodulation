function rand_init(seed)
    % rand_init: 初始化随机数生成器的状态
    %
    % 参数:
    %   seed: 随机数生成器的种子
    %
    % 全局变量:
    %   seed1, seed2, seed3, seed4, seed5: 随机数生成器的状态变量

    global seed1 seed2 seed3 seed4 seed5

    % 设置初始种子
    seed = 89; % 用于初始化线性同余法生成器的种子

    % 计算随机数生成器的状态
    seed1 = mod(171 * seed, 30269);
    seed2 = mod(172 * seed1, 30307);
    seed3 = mod(170 * seed2, 30323);

    % 根据生成器的状态计算其他变量
    k = fix(seed3 / 52774);
    seed5 = 40692 * (seed3 - k * 52774) - k * 3791;
    if seed5 < 0
        seed5 = seed5 + 2147473399;
    end

    k = fix(seed5 / 53668);
    seed4 = 40014 * (seed5 - k * 53668) - k * 12211;
    if seed4 < 0
        seed4 = seed4 + 2147483563;
    end
end
