module HWconstrained

greet() = print("Hello World!")

	using JuMP, NLopt, DataFrames, Ipopt
	using LinearAlgebra

	export data, table_NLopt, table_JuMP

	function data(a=0.5)
		na = 3
		nc = 0
		ns = 4
		nss = ns^2
		p = [1, 1, 1]
		e = [2, 0, 0]
		pi = repeat([1/nss], nss)    #vector of probabilities
		z2 = [0.72, 0.92, 1.12, 1.32]
		z3 = [0.86, 0.96, 1.06, 1.16]
		zJ = [[1.0, i, j] for i in z2 for j in z3]   # for JuMP
		zNL = vcat(transpose(zJ)...) # for NLopt
		return Dict("a"=>a,"na"=>na,"nc"=>nc,"ns"=>ns,"nss"=>nss,"e"=>e,"p"=>p,"zJ"=>zJ, "zNL"=>zNL,"pi"=>pi)
	end

	function max_JuMP(data::Dict, a = 0.5)
		u(x) = -exp(-data["a"] * x)
		m = Model(with_optimizer(Ipopt.Optimizer))
		@variable(m, c >= 0)   #non negative consumption
		@variable(m, x[1:3])   #should also be nonnegative!!
		@NLconstraint(m, c + sum(data["p"] .* (x[1:3] - data["e"])) == 0)
		@NLobjective(m, Max, u(c) + sum(data["pi"] .* u.(data["zJ"] * x[1:3])))
		JuMP.optimize!(m)
		return Dict("obj"=>objective_value(m),"c"=>value(c),"omegas"=>[value(x[i]) for i in 1:length(x)])
		return Dict("obj"=>objective_value(m),"c"=>value(c),"omegas"=>[value(omega[i]) for i in 1:length(omega)])
	end

	function table_JuMP()
		d = DataFrame(a=[0.5;1.0;5.0],c = zeros(3),omega1=zeros(3),omega2=zeros(3),omega3=zeros(3),fval=zeros(3))
		for i in 1:nrow(d)
			xx = max_JuMP(d[i,:a])
			d[i,:c] = xx["c"]
			d[i,:omega1] = xx["omegas"][1]
			d[i,:omega2] = xx["omegas"][2]
			d[i,:omega3] = xx["omegas"][3]
			d[i,:fval] = xx["obj"]
		end
		return d
	end

	function obj(x::Vector,grad::Vector,data::Dict)
		u(x) = -exp(-data["a"] * x)
    	uprime(x) = data["a"] * exp(-data["a"] * x)
    	if length(grad) > 0
        	grad[1] = uprime(x[1])       #gradiant wrt c
        	grad[2:end] = sum(data["pi"] .* data["zNL"] .* uprime.(data["zNL"] * x[2:end]), dims=1)     #gradients wrt ω_i
    	end
    	return u(x[1]) + sum(data["pi"] .* u.(data["zNL"] * x[2:end]))
	end

	function constr(x::Vector,grad::Vector,data::Dict)
		if length(grad) > 0
        grad[1] = 1           #gradient wrt c
        grad[2:end] = data["p"]   #gradient wrt ω_i
    	end
    return x[1] + sum(data["p"] .* (x[2:end] - data["e"]))
	end

	function max_NLopt(a=0.5)
		    d = data(a)
		    opt = Opt(:LD_MMA, 4)
		    lower_bounds!(opt, [0., -Inf, -Inf, -Inf])
		    max_objective!(opt, (x, grad) -> obj(x, grad, d))
		    inequality_constraint!(opt, (x, grad) -> constr(x, grad, d))
		    ftol_rel!(opt, 1e-9)
		    NLopt.optimize(opt,[0, 2, 0, 0])
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
