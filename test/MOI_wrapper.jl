module TestMySDPSolver

import MySDPSolver
using MathOptInterface
using Test

const MOI = MathOptInterface
const MOIU = MOI.Utilities

const OPTIMIZER = MySDPSolver.Optimizer()
MOI.set(OPTIMIZER, MOI.Silent(), true)

# See the docstring of MOI.Test.Config for other arguments.
const CONFIG = MOI.Test.TestConfig(
    # Modify tolerances as necessary.
    atol = 1e-6,
    rtol = 1e-6,
    # Use MOI.LOCALLY_SOLVED for local solvers.
    optimal_status = MOI.OPTIMAL,
    # Pass attributes or MOI functions to `exclude` to skip tests that
    # rely on this functionality.
    # exclude = Any[MOI.VariableName, MOI.delete],
)
const cache = MOIU.UniversalFallback(MOIU.Model{Float64}())
const cached = MOIU.CachingOptimizer(cache, OPTIMIZER)
const bridged = MOI.Bridges.full_bridge_optimizer(cached, Float64)

function test_SolverName()
    @test MOI.get(OPTIMIZER, MOI.SolverName()) == "MySDPSolver"
    return
end

INCLUDED = ["solve_zero_one_with_bounds"]
@testset "Unit" begin
    MOI.Test.unittest(bridged, CONFIG,
    setdiff([d for d in keys(MOI.Test.unittests)], INCLUDED))
end

"""
    runtests()

This function runs all functions in the this Module starting with `test_`.
"""
# function runtests()
#     for name in names(@__MODULE__; all = true)
#         if startswith("$(name)", "test_")
#             @testset "$(name)" begin
#                 getfield(@__MODULE__, name)()
#             end
#         end
#     end
# end

"""
    test_runtests()

This function runs all the tests in MathOptInterface.Test.

Pass arguments to `exclude` to skip tests for functionality that is not
implemented or that your solver doesn't support.
"""
# function test_runtests()
#     MOI.Test.runtests(
#         BRIDGED,
#         CONFIG,
#         exclude = []
#     )
#     return
# end

end

# TestMySDPSolver.runtests()
