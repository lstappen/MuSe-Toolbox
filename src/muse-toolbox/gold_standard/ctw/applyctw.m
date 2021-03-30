function aliGtw = applyctw(inp, m, factor)
% Apply generalized CTW to m sequences of length n.
%
% Input
%   inp       -  sequences, m x n
%    m        -  number of sequences
%   factor    -  factor of how many points to align relative to seq len n
%
% Output
%   ali      -  alignment
%     objs   -  objective value, 1 x nIt
%     its    -  iteration step ids, 1 x nIt
%     P      -  warping path, l x m
%     Vs     -  transformation, 1 x m (cell), di x b


X0s = cellss(1, m);

if isequal(size(X0s), size(inp))
    X0s = inp;
else
    for i = 1 : m
        X0s{1,i} = inp(i,:);
    end
end

%% algorithm parameter
parCca = st('d', .8, 'lams', 0.7);  % 3 0, .8 .6
parGN = st('nItMa', 2, 'inp', 'linear');
parGtw = st('nItMa', 20, 'debg', 'n');

%% feature
Xs = pcas(X0s, st('d', 3, 'cat', 'n')); % project all the three sequences to 3-D  3

%% monotonic basis
ns = cellDim(Xs, 2);
l = round(min(ns) * factor);  %1.1 max
bas = baTems(l, ns, 'pol', [5 .5], 'tan', [5 1 1]); % 2 polynomial and 3 tangent functions

%% utw (initialization)
aliUtw = utw(Xs, bas, []);

%% gtw
aliGtw = gtw(X0s, bas, aliUtw, [], parGtw, parCca, parGN);
