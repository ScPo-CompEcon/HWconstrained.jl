module HWconstrained

greet() = print("Hello World!")

	using JuMP, NLopt, DataFrames, Ipopt
	using LinearAlgebra

	export data, table_NLopt, table_JuMP, max_JuMP, max_NLopt

	function data(a=0.5)
		na = 3
		p = ones(3)
		e = [2.0, 0.0, 0.0]
		z = zeros(16,3)
		z_2 = [0.72, 0.92, 1.12, 1.32]
		z_3 = [0.86, 0.96, 1.06, 1.16]
		z[:,1] .= 1.0
		for i=1:4
		z[1+(4*(i-1)):4+(4*(i-1)), 2] .= z_2[i]
		end
		for j = 1:4
			z[j,3] = z_3[j]
			z[j+4,3] = z_3[j]
			z[j+4*2,3] = z_3[j]
			z[j+4*3,3] = z_3[j]
		end
		ns = length(z_2)
		nss = length(z_2) * length(z_3)
		pi = 1/nss
		nc = 1

        # na number of assets
		# ns number of states per asset
		# nss total number of assets
		# nc number of constraints


		return Dict("a"=>a,"na"=>na,"nc"=>nc,"ns"=>ns,"nss"=>nss,"e"=>e,"p"=>p,"z"=>z,"pi"=>pi)
	end

dat = data()

	function max_JuMP(a=0.5)

		m = Model(with_optimizer(Ipopt.Optimizer))
		@variable(m, omega[1:3])
		@variable(m, c >= 0.0)
		@NLconstraint(m, c + sum(data()["p"][i] * (omega[i] - data()["e"][i]) for i in 1:3) == 0.0)
		@NLobjective(m, Max,
			-exp(-a*c) + data()["pi"] * sum(-exp(-a *
			sum(data()["z"][j,i] * omega[i] for i in 1:3)) for j in 1:data()["nss"]))
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


	## Does not work
	function obj(x::Vector,grad::Vector,data::Dict)
		if length(grad) > 0
			# grad wrt x1
			a = data["a"]
			# grad wrt c
			grad[4] = (-a) * exp(-a * c)
			# grad with respect to omega 1
			grad[1] = (-a) * data["pi"] * sum(data["z"][i,1] * exp(-a *
			sum(data["z"][j,i] * omega[i] for i in 1:3)) for j in 1:data["nss"])

			# grad wrt to omega 2
			grad[2] = (-a) * data["pi"] * sum(data["z"][i,2] * exp(-a *
			sum(data["z"][j,i] * omega[i] for i in 1:3)) for j in 1:data["nss"])

			# grad wrt to omega 3
			grad[3] = (-a) * data["pi"] * sum(data["z"][i,3] * exp(-a *
			sum(data["z"][j,i] * omega[i] for i in 1:3)) for j in 1:data["nss"])

		end
		return (-1) * (-exp(-a*x[4]) + data["pi"] * sum(-exp(-a *
		sum(data["z"][j,i] * x[i] for i in 1:3)) for j in 1:data["nss"]))
	end



	function constr(x::Vector,grad::Vector,data::Dict)
		if length(grad) > 0
			grad[4] = 1
			grad[1] = data["p"][1]
			grad[2] = data["p"][2]
			grad[3] = data["p"][3]

		end
		return x[4] + sum(data["p"][i] * (x[i] - data["e"][i]) for i in 1:3)

	end


	function max_NLopt(a=0.5)
		opt = Opt(:LD_MMA, 4)
		lower_bounds!(opt, [-Inf, -Inf, -Inf, 0.])
		xtol_rel!(opt, 1e-4)
		min_objective!(opt,(x,grad) -> obj(x,grad,data()))
		inequality_constraint!(opt, (x,grad) -> constr(x,grad,data()))
		ftol_rel!(opt,1e-9)
		vector_init = vcat(data()["e"], 1.0)
		NLopt.optimize(opt, vector_init)
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
