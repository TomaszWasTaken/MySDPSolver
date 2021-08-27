#=
    A very simple example of a semidefinite program
    in standard form. The problem is given by
    min tr(CX)
    st  tr(AX) = b
        X ⪰ 0
=#

using MySDPSolver
using JuMP
using LinearAlgebra

# Data
A = [1.0 0.0
     0.0 0.0]
b = 1.0
C = [2.0 1.0
     1.0 2.0]
# Set the model
model = Model(MySDPSolver.Optimizer)
@variable(model, X[1:2,1:2], PSD)
@constraint(model, tr(A*X) == b)
@objective(model, Min, tr(C*X))
optimize!(model)
