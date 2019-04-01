using HWconstrained
using Test

@testset "HWconstrained.jl" begin
	@testset "testing components" begin
		z = data["z"]

		@testset "tests gradient of objective function" begin
			        a = 0.5
                    obj(x) = exp(-a * x[1]) + sum(exp(-a * z[i]' * x[2:4]) for i in 1:16)/16
					gradient = [-a * exp(-a * x[1]),
					-sum(a * exp(-a * z[i][2] * sum(z[i]' * x[2:4])) for i in 1:16)/16,
					-sum(a * exp(-a * z[i][3] * sum(z[i]' * x[2:4])) for i in 1:16)/16,
					-sum(a * exp(-a * z[i][4] * sum(z[i]' * x[2:4])) for i in 1:16)/16]
					@test gradient =

		end


		@testset "tests gradient of constraint function" begin
			cons = x[1] + sum(x[2:4] - data["e"])
			gradcon = ones(4)
			@test gradcon =

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
