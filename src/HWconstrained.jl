module HWconstrained

greet() = print("Hello World!")

	using JuMP, NLopt, DataFrames, Ipopt
	using LinearAlgebra

	export data, table_NLopt, table_JuMP

	function data(a=0.5)
        na = 3
		nc = 1
		ns = 4
		nss = 16
		e = [2,0,0]
		p = [1,1,1]
		z_2 = [0.72,0.92,1.12,1.32]
		z_3 = [0.86,0.96,1.06,1.16]
		z = [[1.0,z_2[j],z_3[k]] for j in 1:4 for k in 1:4]
		pi = 1/16*ones(1,16)
		return Dict("a"=>a,"na"=>na,"nc"=>nc,"ns"=>ns,"nss"=>nss,"e"=>e,"p"=>p,"z"=>z,"pi"=>pi)
	end

	function maxJuMP(a=0.5)
	        d = data(a)
			n = Model(with_optimizer(Ipopt.Optimizer))
			@variable(n, omega[1:3])
			@variable(n, c)
			@NLobjective(n, Min, exp(-a*c)+sum(d["pi"][i]*exp((-a)*sum(d["z"][i][j]'*omega[j] for j in 1:d["na"])) for i in 1:d["nss"]))
			@constraint(n, c + d["p"]'*(omega - d["e"]) <= 0)
			JuMP.optimize!(n)
	  	return Dict("obj"=>objective_value(n),"c"=>value(c),"omegas"=>[value(omega[i]) for i in 1:length(omega)])
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

	function obj(x::Vector,data::Dict)
		a = data["a"]
		z = data["z"]
		if length(grad)>0
			  grad[1] = sum(data["pi"][i]*(-a)*exp((-a)*z[i][1]*sum(z[i]'*x[1:3])) for i in 1:data["nss"])
			  grad[2] = sum(data["pi"][i]*(-a)*exp((-a)*z[i][2]*sum(z[i]'*x[1:3])) for i in 1:data["nss"])
			  grad[3] = sum(data["pi"][i]*(-a)*exp((-a)*z[i][3]*sum(z[i]'*x[1:3])) for i in 1:data["nss"])
			  grad[4] = (-a)*exp.(-a*x[4])
		  end
		return (exp(-a*x[4])+sum(data["pi"][i]*exp((-a)*sum(z[i]'*x[1:3])) for i in 1:data["nss"]))
	  end


	function constr(x::Vector,grad::Vector,data::Dict)
		if length(grad)>0
			   grad[1] = data["p"][1]
			   grad[2] = data["p"][2]
			   grad[3] = data["p"][3]
			   grad[4] = 1
	   end
	   return x[4] + data["p"]'*(x[1:3] - data["e"])
	end







	function max_NLopt(a=0.5)
		opt = Opt(:LD_MMA, 4)
		d = HWconstrained.data(a)
		objE(x,g;data = d) = HWconstrained.obj(x,g,data)
		constrE(x,g;data = d) = HWconstrained.constr(x,g,data)
		NLopt.min_objective!(opt,(x,g)->objE(x,g))
		xtol_rel!(opt,1e-10)
		lower_bounds!(opt, [-Inf,-Inf,-Inf,0])
		inequality_constraint!(opt,(x,g)->constrE(x,g))
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
