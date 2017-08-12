function [x,f] = StochGraDes(funObj,x0,options)

alpha = 0.002; 
x = x0;
[f_old,grad] = funObj(x);

x = x - alpha * grad; 
[f,grad] = funObj(x); 
end