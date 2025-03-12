function g = dnorm(x,mult,sigma)

g = 1/(sqrt(2*pi)*sigma)*exp(-(x-mult).^2/(2*sigma^2));