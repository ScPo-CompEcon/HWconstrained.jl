
module HWconstrained

greet() = print("Hello World!")

	using JuMP, NLopt, DataFrames, Ipopt
	using LinearAlgebra

	export data, table_NLopt, table_JuMP

	function data(a=0.5)
		p = [1,1,1]
		na = 3
		ns = 4
		nss = 16
		z2 = [0.72, 0.92, 1.12, 1.32]
		z3 = [0.86, 0.96, 1.06, 1.16]
		z = [[1.0,z2[j],z3[k]] for j in 1:4 for k in 1:4]
        e = [2,0,0]
        nc = 1
		pi = 1/16

		return Dict("a"=>a,"na"=>na,"nc"=>nc,"ns"=>ns,"nss"=>nss,"e"=>e,"p"=>p,"z"=>z,"pi"=>pi)
	end


	function max_JuMP(a=0.5)
		d = HWconstrained.data(a)
		z = d["z"]
		nss = d["nss"]
		pi = d["pi"]
		p = d["p"]
		e = d["e"]
		m = Model(with_optimizer(Ipopt.Optimizer))
		@variable(m, c >= 0)
		@variable(m, omega[1:3])
		@NLobjective(m, Max, -exp(-a*c) + pi*sum(-exp(-(a*sum(z[s][i]*omega[i] for i in 1:3))) for s in 1:nss))
		@constraint(m, c + p'*(omega - e) <= 0)
		@constraint(m, c + p'*(omega - e) >= 0)
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



	function u(x,a)
		return -exp(-(a*x))
	end

	function u_prime(x,a)
		return a*exp(-(a*x))
	end




	function obj(x::Vector,grad::Vector,data::Dict)
		nss = data["nss"]
		z = data["z"]
		a = data["a"]
		pi = data["pi"]
		if length(grad) > 0
			grad[1] = (HWconstrained.u_prime(x[1],a))
			grad[2] = pi*sum(HWconstrained.u_prime(sum(z[s]'*x[2:4]),a)*z[s][1] for s in 1:nss)
			grad[3] = pi*sum(HWconstrained.u_prime(sum(z[s]'*x[2:4]),a)*z[s][2] for s in 1:nss)
			grad[4] = pi*sum(HWconstrained.u_prime(sum(z[s]'*x[2:4]),a)*z[s][3] for s in 1:nss)
		end
		return (HWconstrained.u(x[1],a) + pi*sum(HWconstrained.u(sum(z[s]'*x[2:4]),a) for s in 1:nss))
	end

	function constr(x::Vector,grad::Vector,data::Dict)
		e = data["e"]
		p = data["p"]
		if length(grad) > 0
			grad[1] = 1
			grad[2] = p[1]
			grad[3] = p[2]
			grad[4] = p[3]
		end
		return (x[1] + p'*(x[2:4] - e))
	end

	function max_NLopt(a=0.5)
		opt = Opt(:LD_MMA, 4)
		d = HWconstrained.data(a)
		objE(x,g;data = d) = obj(x,g,data)
		constrE(x,g;data = d) = constr(x,g,data)
		NLopt.max_objective!(opt,(x,g)->objE(x,g))
		xtol_rel!(opt,1e-10)
		lower_bounds!(opt, [0.,-Inf,-Inf,-Inf])
		NLopt.inequality_constraint!(opt,(x,g)->constrE(x,g))
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
