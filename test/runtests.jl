using HWconstrained
using Test, ForwardDiff

@testset "HWconstrained.jl" begin
	@testset "testing components" begin
		d = data()
		u(x) = -exp(-d["a"] * x)
		grad = zeros(4)

		@testset "tests gradient of objective function" begin
			objfun(x) = u(x[1]) + sum(d["pi"] .* u.(d["z"] * x[2:end]))
			t = ForwardDiff.gradient(objfun, ones(4))
			obj(ones(4), grad, d)
			@test grad == t
		end


		@testset "tests gradient of constraint function" begin
			consfun(x) = x[1] + sum(d["p"] .* (x[2:end] - d["e"]))
			t = ForwardDiff.gradient(consfun, ones(4))
			constr(ones(4), grad, d)
			@test grad == t
		end
	end

	@testset "testing result of both maximization methods" begin

		truth = HWconstrained.DataFrame(a=[0.5;1.0;5.0],
			                            c = [1.008;1.004;1.0008],
			                            omega1=[-1.41237;-0.20618;0.758763],
			                            omega2=[0.801455;0.400728;0.0801455],
			                            omega3=[1.60291;0.801455;0.160291],
			                            fval=[-1.20821;-0.732819;-0.013422])
		tol2 = 1e-2

		@testset "checking result of NLopt maximization" begin

			t1 = table_NLopt()
			for c in names(truth)
				@test all(maximum.(abs.(t1[c].-truth[c])) .< tol2)
			end
		end


		@testset "checking result of NLopt maximization" begin
			t1 = table_JuMP()
			for c in names(truth)
				@test all(maximum.(abs.(t1[c].-truth[c])) .< tol2)
			end
		end
	end

end
