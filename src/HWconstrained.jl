module HWconstrained

greet() = print("Hello World!")

	using JuMP, NLopt, DataFrames, Ipopt
	using LinearAlgebra

	export data, table_NLopt, table_JuMP

	function data(a=0.5)
		z2 = [0.72, 0.92, 1.12, 1.32]
		z3 = [0.86, 0.96, 1.06, 1.16]
		z = [[1.0, z2[i],z3[j]] for i in 1:4 for j in 1:4]
		e = [2, 0, 0]

		return Dict("e"=>e,"z"=>z)
	end


	function max_JuMP(a=0.5)
		d = data(a)
		m = Model(with_optimizer(Ipopt.Optimizer))
		@variable(m, 0 <= c)
		@variable(m, omega[1:3])
		@NLconstraint(m, 0 == c + sum(omega[i] - d["e"][i] for i in 1:3))
	    @NLobjective(m, Max, -exp(-a*c) + sum(-exp(-a*sum(d["z"][j][i] * omega[i] for i in 1:3)) / 16 for j in 1:16))
		JuMP.optimize!(m)

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
		a = data["a"]
		z = data["z"]
		if length(grad) > 0
			grad[1] = -a * exp(-a * x[1])
			grad[2] = -sum(a * exp(-a * z[i][2] * sum(z[i]' * x[2:4])) for i in 1:16)/16
			grad[3] = -sum(a * exp(-a * z[i][3] * sum(z[i]' * x[2:4])) for i in 1:16)/16
			grad[4] = -sum(a * exp(-a * z[i][4] * sum(z[i]' * x[2:4])) for i in 1:16)/16
		end
		return exp(-a * x[1]) + sum(exp(-a * z[i]' * x[2:4]) for i in 1:16)/16

	end

	function constr(x::Vector,grad::Vector,data::Dict)
		if length(grad) > 0
			grad[1:4] = 1
		end
		return x[1] + sum(x[2:4] - data["e"])

	end

	function max_NLopt(a=0.5)
		d = data(a)
		opt = Opt(:LD_MMA, 4)
		lower_bounds!(opt, [0., -Inf, -Inf, -Inf])
		xtol_rel!(opt,1e-2)
		min_objective!(opt, (x,grad) -> obj(x, grad, d))
		inequality_constraint!(opt, (x, grad) -> constr(x, grad, d))
		return NLopt.optimize(opt, zeros(4))

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
