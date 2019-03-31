using HWconstrained
using Test

@testset "HWconstrained.jl" begin
	@testset "testing components" begin

		@testset "tests gradient of objective function" begin

			truth = HWconstrained.DataFrame(a=[0.5;1.0;5.0],
											c = [1.008;1.004;1.0008],
											omega1=[-1.41237;-0.20618;0.758763],
											omega2=[0.801455;0.400728;0.0801455],
											omega3=[1.60291;0.801455;0.160291],
											fval=[-1.20821;-0.732819;-0.013422])
			x = zeros(3,4)
			x[:,1] = truth[:omega1]
			x[:,2] = truth[:omega2]
			x[:,3] = truth[:omega3]
			x[:,4] = truth[:c]
			grad = rand(size(x,2))
			testvec = deepcopy(grad)
			d = data()
			# Extract gradient
			grad! = (x,grad) -> HWconstrained.obj(x,grad,d)

			r = grad!(x[1,:],grad)
			# Cannot test gradient is zero since constrained regression
			@test grad != testvec

		end


		@testset "tests gradient of constraint function" begin
			truth = HWconstrained.DataFrame(a=[0.5;1.0;5.0],
											c = [1.008;1.004;1.0008],
											omega1=[-1.41237;-0.20618;0.758763],
											omega2=[0.801455;0.400728;0.0801455],
											omega3=[1.60291;0.801455;0.160291],
											fval=[-1.20821;-0.732819;-0.013422])
			x = zeros(3,4)
			x[:,1] = truth[:omega1]
			x[:,2] = truth[:omega2]
			x[:,3] = truth[:omega3]
			x[:,4] = truth[:c]
			grad = rand(size(x,2))
			testvec = deepcopy(grad)
			d = data()
			# Extract gradient
			gradconstr! = (x,grad) -> HWconstrained.constr(x,grad,d)

			r = gradconstr!(x[1,:],grad)
			# Cannot test whether the gradient is zero since it is a constraint regression
			@test grad != testvec





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
			tol2 = 1e-1
			t1 = table_NLopt()
			for c in names(truth)
				@test all(maximum.(abs.(t1[c].-truth[c])) .< tol2)
			end
		end


		@testset "checking result of NLopt maximization" begin
			tol2 = 1e-1
			t1 = table_JuMP()
			for c in names(truth)
				@test all(maximum.(abs.(t1[c].-truth[c])) .< tol2)
			end
		end
	end

end
