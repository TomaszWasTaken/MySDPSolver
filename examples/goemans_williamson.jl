using MySDPSolver, JuMP, LinearAlgebra          # -> opti
using LightGraphs, GraphPlot, Gadfly, Colors    # -> graphs + plots
using Random, Distributions                     # -> random

#=
Simple example illustrating the Goemans-Williamson approach for the max-cut
as in http://www.cs.toronto.edu/~toni/Courses/Proofs-SOS-2018/Lectures/maxcut.pdf
=#
function GW(g::SimpleGraph)
    # Get the laplacian of g
    L = laplacian_matrix(g)
    n = size(L, 1)

    # Set the SDP relaxation
    model = Model(() -> MySDPSolver.Optimizer(verbose=false))
    @variable(model, X[1:n,1:n], PSD)
    @objective(model, Max, 0.25tr(L*X))
    @constraint(model, diag(X) .== ones(Float64, n))
    optimize!(model)

    # Upper Cholesky factor
    u = cholesky(value.(X)).U
    # Random hyperplane
    r = rand(Normal(), n)
    # Solution vector
    x = ones(Float64, n)

    # Randomized rounding
    for i = 1:n
        if dot(u[:,i], r) < 0.0
            x[i] *= -1.0
        end
    end

    return x
end

function main()
    # Compute the approx. max-cut
    g = smallgraph(:karate)  # a simple graph
    x = GW(g)
    # Color the nodes
    val2color(x_i) = x_i < 0.0 ? 2 : 1  # -1: red, 1:blue
    membership = val2color.(x)
    nodecolor = [colorant"blue", colorant"red"]
    nodefillc = nodecolor[membership]
    # Plot the graph and the partition
    gplot(g, nodefillc=nodefillc)
end

main()
