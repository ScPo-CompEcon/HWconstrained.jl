using HWconstrained
using Test, Calculus

@testset "HWconstrained.jl" begin
	@testset "testing components" begin

			d = data()
			u(x) = -exp(-d["a"]*x)
			grad = zeros(4)

		@testset "tests gradient of objective function" begin
				objf(x) = u(x[1]) + sum(d["pi"] .* u.(d["zNL"] * x[2:end]))
                objtest = Calculus.gradient(objf, ones(4))
                obj(ones(4), grad, d)
                @test objtest == grad
		end


		@testset "tests gradient of constraint function" begin
                conf(x) =    x[1] + sum(data["p"] .* (x[2:end] - data["e"]))
                contest = Calculus.gradient(conf, ones(4))
				constr(ones(4), grad, d)
                @test contest == grad
		end
	end

	@testset "testing result of both maximization methods" begin

		truth = HWconstrained.DataFrame(a=[0.5;1.0;5.0],
			                            c = [1.008;1.004;1.0008],
			                            omega1=[-1.41237;-0.20618;0.758763],
			                            omega2=[0.801455;0.400728;0.0801455],
			                            omega3=[1.60291;0.801455;0.160291],
			                            fval=[-1.20821;-0.732819;-0.013422])

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
