module HWconstrained

greet() = print("Hello World!")

	using JuMP, NLopt, DataFrames, Ipopt
	using LinearAlgebra

	export data, table_NLopt, table_JuMP

function data(a=0.5)
        a = a
		na = 3
		nc = 4 # Budget constraint + 3 nonnegativity constraints
		ns = 4
		nss = 16
		z1 = ones(4)
		z2 = [0.72, 0.92, 1.12, 1.32]
		z3 = [0.86, 0.96, 1.06, 1.16]
		z = [(1, i, j) for i in z2 for j in z3]  # All possible payoffs
		p = ones(3)
		pi = repeat([1/16], nss)
		e = [2, 0, 0]

		return Dict("a"=>a,"na"=>na,"nc"=>nc,"ns"=>ns,"nss"=>nss,"e"=>e,"p"=>p,"z"=>z,"pi"=>pi)
end


	function max_JuMP(a=0.5)
		d = data(a)
		m = Model(with_optimizer(Ipopt.Optimizer))

		@variable(m, c>=0)
		@variable(m, x[1:3])
		@NLconstraint(m, c + sum(d["p"][i] * (x[i] - d["e"][i]) for i in 1:3) == 0 )
		@NLobjective(m, Max, -exp(-a * c) + sum(-exp(-a * sum(d["z"][j][i] * x[i] for i in 1:3)) * d["pi"][j] for j in 1:16))
		JuMP.optimize!(m)
		return Dict("obj"=>objective_value(m),"c"=>value(c),"omegas"=>[value(x[i]) for i in 1:length(x)])
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





function obj(x::Vector, grad::Vector, data::Dict)
    a = data["a"]
    z = data["z"]
	u(x) = -exp(-a * x)
   if length(grad) > 0
        grad[1] = -a * u(x[1], data)
    for i in 1:3
            grad[i+1] = -a .* sum(data["pi"] .* z[:, i] .* u.(z * x[2:end], Ref(data)))
    end
end
   return u(x[1], data) + sum(data["pi"] .* u.(z * x[2:end], Ref(data)))
end


function constr(x::Vector,grad::Vector,data::Dict)
    if length(grad) > 0
	        grad[1] = 1
	        grad[2:end] = d["p"]
	 end
	return x[1] + sum(d["p"] .* (x[2:end] .- d["e"]))
end

function max_NLopt(a=0.5)
    d = data(a)
    opt = Opt(:LD_MMA, 4)
    lower_bounds!(opt, [0, -Inf, -Inf, -Inf]) # As cannot sell more then initial endowment
    max_objective!(opt, (x,g) -> obj(x,g, d))
    inequality_constraint!(opt, (x, g) -> constr(x, g, d))
    ftol_rel!(opt, 1e-9)
    NLopt.optimize(opt, [0,2,0,0])
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
