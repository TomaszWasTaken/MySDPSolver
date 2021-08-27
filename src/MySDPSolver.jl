module MySDPSolver
    # Import the necessary packages.
    using LinearAlgebra, SparseArrays, MKLSparse
    using Printf, Logging
    using MathOptInterface

    # ! BlasInt: Depending on how BLAS is build,
    # arrays can be indexed as Int32 or Int64.
    # Using the wrong type causes a segmentation fault,
    # Hence we need to retrieve this info.
    import LinearAlgebra: BlasInt

    # Include the files.
    include("my_problem.jl")
    include("my_solver.jl")
    include("utils.jl")

    const MOI = MathOptInterface
    const MOIU = MOI.Utilities
    include("MOI_wrapper.jl")

end # module
