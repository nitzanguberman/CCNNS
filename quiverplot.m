function []=quiverplot(A,varargin)
quiver(real(A),imag(A),varargin{:});
axis off;
end