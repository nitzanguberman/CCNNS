function S = logsumexp(X, dim)
%LOGSUM executes: S = log(sum(exp(X)), along dim in a precision-aware
% manner.
%
%   Arguments:
%       X - a matrix in log space.
%       dim - the dimension along which summation is performed,
%             defaults to 1, i.e. rows.
if ~exist('dim','var') || isempty(dim) dim = 1; end
c = size(X, dim); rep_dims = ones(1,length(size(X))); rep_dims(dim) = c;
m = max(X, [], dim);
M = repmat(m, rep_dims);
S = squeeze(m + log(sum(exp(X-M),dim)));
S(isnan(S)) = -inf; %only reason for a NaN is -inf - -inf
end