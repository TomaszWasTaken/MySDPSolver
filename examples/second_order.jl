#=
    A simple conic problem using the second-order cone.

    The model is defined as
    min ‖u - u0‖₂
    st  p'u = q

    The conic formulation we will solve is given by
    min t
    st  p'u = q
        ‖u- u0‖₂ ≤ t  (second-order cone)

    Since second-order programming is a special case of SDP,
    it can be reformulated accordingly. This example illustrates
    the reformulation capabilities of JuMP/MathOptInterface.
    Indeed, second-order cones were not implemented in MySDPSolver,
    but MOI can handle them.
=#

using MySDPSolver, JuMP, LinearAlgebra

# Some data
u0 = ones(Float64, 10)
p = ones(Float64, 10)
q = 1.0

# Declare the model
model = Model(MySDPSolver.Optimizer)

# Add the variables and the objective
@variable(model, u[1:10])
@variable(model, t)
@objective(model, Min, t)

# Add the conic constraint, i.e. ‖u-u0‖₂ ≤ t
@constraint(model, c1, [t, (u - u0)...] in SecondOrderCone())  # see 'splatting' for the '...' syntax
@constraint(model, c2, u'*p == q)

# Call the solver
optimize!(model)

# Retrieve the solutions
println("t:")
display(value(t))
println("u: ")
display(value.(u))

# We can get the dual solutions as well
println("dual solutions:")
display(dual(c1))
display(dual(c2))

println("###############################################")
