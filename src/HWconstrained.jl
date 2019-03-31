module HWconstrained

using JuMP, NLopt, DataFrames, Ipopt
using LinearAlgebra
export data, table_NLopt, table_JuMP

### Question 1
function data(a=0.5)
n=3 
p=[1, 1, 1]
e=[2, 0, 0]
s1=s2=4
z1=[1, 1, 1, 1]
z2=[0.72, 0.92, 1.12, 1.32]
z3=[0.86, 0.96, 1.06, 1.16]
z=[[1, i, j] for i in z2 for j in z3]
z = vcat(z'...)
pi = repeat([1/16], 16)
a=0.5
na=3
nc=4
ns=4
nss=16

return Dict("a"=>a,"na"=>na,"nc"=>nc,"ns"=>ns,"nss"=>nss,"e"=>e,"p"=>p,"z"=>z,"pi"=>pi)
end
#end

d=data()

####  Question 2

function obj(x::Vector,grad::Vector,data::Dict)
    A = data["a"]
    Z = data["z"]
    pi = data["pi"]
    if length(grad) > 0
        grad[1] = A*exp.(-A*x[1])
        for i in 1:3
        grad[i+1] = sum(pi .* Z[:,i] .*A.*exp.(-A.*Z*x[i+1]))
        end
    end
    return -exp.(-A*x[1])+ sum(pi.*-exp.(-A*Z*x[2:4]))
end
obj(ones(4), zeros(4), d)

function constr(x::Vector,grad::Vector,data::Dict)
    if length(grad) > 0
        grad[1] = d["a"]*exp(-d["a"]*x[1])
        grad[2:end] = d["p"]
    end
    return x[1] + sum(d["p"].*(x[2:end].-d["e"]))
end    
    
constr(ones(4), zeros(4), d) # keep track of # function evaluations

function max_NLopt(a=0.5)
d = data(a)
e= d["e"]
optimum = Opt(:LD_MMA, 4)
lower_bounds!(optimum, [0., -Inf, -Inf, -Inf])   
max_objective!(optimum, (x, g)->obj(x, g, d), 1e-8)
inequality_constraint!(optimum, (x, g)->constr(x, g, d), 1e-8)
ftol_rel!(optimum, 1e-8)
NLopt.optimize(optimum)
end

function table_NLopt()
d = DataFrame(a=[0.5;1.0;5.0],c = zeros(3),omega1=zeros(3),omega2=zeros(3),omega3=zeros(3),fval=zeros(3))
for i in 1:nrow(d)
xx = max_NLopt(d[i,:a])
for j in 2:ncol(d)-1
d[i,j] = xx[2][j-1]
end
d[i,end] = xx[1]
end
return d
end



end # module
