#=
  MathOptInterface wrapper.

  The wrapper allows to access the solver through JuMP.
  This implementation is greatly inspired by the wrapper for
  the SDPA solver since both use a similar structure.
  The latter can be found at:
  https://github.com/jump-dev/SDPA.jl/blob/master/src/MOI_wrapper.jl
=#


const AFFEQ = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}
const SupportedSets = Union{MOI.Nonnegatives, MOI.PositiveSemidefiniteConeTriangle}

#=
  Structure describing the optimization problem.
  It stores the information about the variables as well
  as some solver's parameters. The numerical data
  for the problem are stored in the 'problem' inner_problem structure
  for convenience.
=#
mutable struct Optimizer <: MOI.AbstractOptimizer
    objconstant::Float64
    objsign::Int
    blockdims::Vector{Int}
    varmap::Vector{Tuple{Int, Int, Int}} # Variable Index vi -> blk, i, j
    b::Vector{Float64}
    problem::Union{Nothing, inner_problem}
    solve_time::Float64
    silent::Bool
    options::Dict{Symbol, Any}
    function Optimizer(; kwargs...)
		optimizer = new(
            zero(Float64), 1, Int[], Tuple{Int, Int, Int}[], Float64[],
            nothing, NaN, false, Dict{Symbol, Any}())
		for (key, value) in kwargs
			MOI.set(optimizer, MOI.RawParameter(key), value)
		end
		return optimizer
    end
end

# Return the name of the solver.
MOI.get(::Optimizer, ::MOI.SolverName) = "MySDPSolver"

# Return the inner problem structure.
MOI.get(opt::Optimizer, ::MOI.RawSolver) = opt.problem

#=
  This part deals with the solver's parameters.
  The workflow consists of writing three functions:
    (1): MOI.supports: boolean function telling MOI if the solver
	                   supports the wanted parameter.
	(2): MOI.get: a getter for the wanted param (if supported).
	(3): MOI.set: a setter (note that some parameters cannot be set).
=#

const SUPPORTED_RAW_PARAMS = [:maxIter, :ε_relgap, :ε_φ, :verbose]

# MOI.RawParameter: a solver-specific attribute.
function MOI.supports(optimizer::Optimizer, param::MOI.RawParameter)
	return param.name in SUPPORTED_RAW_PARAMS
end

function MOI.get(optimizer::Optimizer, param::MOI.RawParameter)
	if haskey(optimizer.options, param.name)
		return optimizer.options[param.name]
	else
		println("ERROR: $(param.name) is not a valid parameter name.")
	end
end

function MOI.set(optimizer::Optimizer, param::MOI.RawParameter, value)
	if !MOI.supports(optimizer, param)
		throw(MOI.UnsupportedAttribute(param))
	end
	optimizer.options[param.name] = value
end

# MOI.Silent: verbosity of the solver. If set to 'true', no info displayed.
MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
	optimizer.silent = value
end

MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent



# Base.hash(p, u::UInt64) = hash(convert(Int32, p), u)

# const RAW_STATUS = Dict(
#     noINFO     => "The iteration has exceeded the maxIteration and stopped with no informationon the primal feasibility and the dual feasibility.",
#     pdOPT      => "The normal termination yielding both primal and dual approximate optimal solutions.",
#     pFEAS      => "The primal problem got feasible but the iteration has exceeded the maxIteration and stopped.",
#     dFEAS      => "The dual problem got feasible but the iteration has exceeded the maxIteration and stopped.",
#     pdFEAS     => "Both primal problem and the dual problem got feasible, but the iterationhas exceeded the maxIteration and stopped.",
#     pdINF      => "At least one of the primal problem and the dual problem is expected to be infeasible.",
#     pFEAS_dINF => "The primal problem has become feasible but the dual problem is expected to be infeasible.",
#     pINF_dFEAS => "The dual problem has become feasible but the primal problem is expected to be infeasible.",
#     pUNBD      => "The primal problem is expected to be unbounded.",
#     dUNBD => "The dual problem is expected to be unbounded.")


function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    if optimizer.problem === nothing
        return "`MOI.optimize!` not called."
    end
	return RAW_STATUS[getPhaseValue(optimizer.problem)]
end

# MOI.SolveTime: timing of the solver.
function MOI.get(optimizer::Optimizer, ::MOI.SolveTime)
	return optimizer.solve_time
end

#=
  is_empty checks if the optimizer is empty. An empty
  optimizer has no variables, constraints or model
  attributes, however it can have optimizer attributes.
  Indeed, we can create a model with none of the above
  and set some parameters.
  Ex: model = Model(MySDPSolver.Optimizer) # empty optimizer
      MOI.set(model, MOI.Silent(), true)   # still an empty optimizer
=#
function MOI.is_empty(optimizer::Optimizer)
    return iszero(optimizer.objconstant) &&
        optimizer.objsign == 1 &&
        isempty(optimizer.blockdims) &&
        isempty(optimizer.varmap) &&
        isempty(optimizer.b)
end

# empty! effectively empties the optimizer structure.
function MOI.empty!(optimizer::Optimizer)
    optimizer.objconstant = zero(Float64)
    optimizer.objsign = 1
    empty!(optimizer.blockdims) # sets a Vector to [] in Julia
    empty!(optimizer.varmap)
    empty!(optimizer.b)
    optimizer.problem = nothing
end

function MOI.supports(
    optimizer::Optimizer,
    ::Union{MOI.ObjectiveSense,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}})
    return true
end

MOI.supports_add_constrained_variables(::Optimizer, ::Type{MOI.Reals}) = false


MOI.supports_add_constrained_variables(::Optimizer, ::Type{<:SupportedSets}) = true
function MOI.supports_constraint(
    ::Optimizer, ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{MOI.EqualTo{Float64}})
    return true
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike; kws...)
    return MOIU.automatic_copy_to(dest, src; kws...)
end

MOIU.supports_allocate_load(::Optimizer, copy_names::Bool) = !copy_names

function MOIU.allocate(optimizer::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    # To be sure that it is done before load(optimizer, ::ObjectiveFunction, ...), we do it in allocate
    optimizer.objsign = sense == MOI.MIN_SENSE ? -1 : 1
end

function MOIU.allocate(::Optimizer, ::MOI.ObjectiveFunction, ::MOI.ScalarAffineFunction) end

function MOIU.load(::Optimizer, ::MOI.ObjectiveSense, ::MOI.OptimizationSense) end

#=
  Utility function returning the block as well as the indices
  within that block (blk, i, j) of a variable indexed by
  vi by MOI.
=#
varmap(optimizer::Optimizer, vi::MOI.VariableIndex) = optimizer.varmap[vi.value]

# Loads objective coefficient α * vi
function load_objective_term!(optimizer::Optimizer, α, vi::MOI.VariableIndex)
    blk, i, j = varmap(optimizer, vi)
    coef = optimizer.objsign * α
    if i != j
        coef /= sqrt(2.0)
    end
	# println("blk, i, j, coef: ", blk,", ", i,", ",j,", ", coef)

	# push!(optimizer.problem.Data.I_obj, i)
	# push!(optimizer.problem.Data.J_obj, j)
	# push!(optimizer.problem.Data.V_obj, coef)
	# inputElement(optimizer.problem, 0, blk, i, j, float(coef), false)
    # in SDP format, it is max and in MPB Conic format it is min
    # inputElement(optimizer.problem, 0, blk, i, j, float(coef), false)
end

function MOIU.load(optimizer::Optimizer, ::MOI.ObjectiveFunction, f::MOI.ScalarAffineFunction)
	# f is of type MOI.ScalarAffineFunction. We put it in the canonical
	# form, which is represented as a triplet (x, terms, constant) where
	# x is a vector of MOI.VariableIndex, coeff are the coefficient and constant
	# is the independant term.
    obj = MOIU.canonical(f)
    optimizer.objconstant = f.constant # extract the constant

	# Preallocate the vectors for the temp. variables.
	I_s = [Int64[] for i in 1:length(optimizer.blockdims)]   # row indices
	J_s = [Int64[] for i in 1:length(optimizer.blockdims)]   # col indices
	V_s = [Float64[] for i in 1:length(optimizer.blockdims)] # val indices

    for t in obj.terms
        if !iszero(t.coefficient)
			blk, i, j = varmap(optimizer, t.variable_index)
            coef = t.coefficient
			# The canonical repr. sums the same variables,
			# ex. for symmetric matrix var., X[1,2] == X[2,1],
			# and since only one is kept, the coefficient will be doubled.
            if i != j
                coef /= 2
            end
			# Push the element to the corresponding entry
			# in the corresponding block.
			push!(I_s[blk], i)
			push!(J_s[blk], j)
			push!(V_s[blk], coef)
        end
    end
	# Vectorize the matrices and update the inner problem in the Optimizer.
	C_temp = [sparse(I_s[k], J_s[k], V_s[k], abs(optimizer.blockdims[k]), abs(optimizer.blockdims[k])) for k in 1:length(optimizer.blockdims)]
	row_temp = [(optimizer.blockdims[k] > 0) ? sparsevec(svec_lower(C_temp[k])) : sparsevec(diag(C_temp[k])) for k in 1:length(optimizer.blockdims)]
	optimizer.problem.normC = max(1.0, maximum(norm.(row_temp)))
	optimizer.problem.C = -optimizer.objsign*vcat(row_temp...)
	I_s = nothing # Help garbage-collect these
	J_s = nothing
	V_s = nothing
	C_temp = nothing
	row_temp = nothing
end

# Add new blocks to the varmap. If the block is diagonally-constrained,
# the size is negative.
function new_block(optimizer::Optimizer, set::MOI.Nonnegatives)
    push!(optimizer.blockdims, -MOI.dimension(set)) # dimension(set) = nb of vars.
    blk = length(optimizer.blockdims)
    for i in 1:MOI.dimension(set)
        push!(optimizer.varmap, (blk, i, i))
    end
end

function new_block(optimizer::Optimizer, set::MOI.PositiveSemidefiniteConeTriangle)
    push!(optimizer.blockdims, set.side_dimension) # dimension(set) = nxn
    blk = length(optimizer.blockdims)
    for i in 1:set.side_dimension
        for j in 1:i
            push!(optimizer.varmap, (blk, i, j))
        end
    end
end

function MOIU.allocate_constrained_variables(optimizer::Optimizer,
                                             set::SupportedSets)
    offset = length(optimizer.varmap)
    new_block(optimizer, set)
    ci = MOI.ConstraintIndex{MOI.VectorOfVariables, typeof(set)}(offset + 1)
    return [MOI.VariableIndex(i) for i in offset .+ (1:MOI.dimension(set))], ci
end

function MOIU.load_constrained_variables(
    optimizer::Optimizer, vis::Vector{MOI.VariableIndex},
    ci::MOI.ConstraintIndex{MOI.VectorOfVariables},
    set::SupportedSets)
end

function MOIU.load_variables(optimizer::Optimizer, nvars)
    @assert nvars == length(optimizer.varmap)
	N = sum((optimizer.blockdims[k] > 0) ? (optimizer.blockdims[k])*(optimizer.blockdims[k]+1)÷2 : abs(optimizer.blockdims[k]) for k in 1:length(optimizer.blockdims))
    dummy = isempty(optimizer.b)
    if dummy
        optimizer.b = [one(Float64)]
        optimizer.blockdims = [optimizer.blockdims; -1]
    end
	# Instantiate inner_problem with correct structure and data set to 0.
    optimizer.problem = inner_problem(optimizer.blockdims,
	                                  spzeros(Float64, N, length(optimizer.b)),
									  -optimizer.b,
									  Vector{Float64}(undef, N))
	optimizer.problem.normsA = zeros(Float64, length(optimizer.blockdims))

end

function MOIU.allocate_constraint(optimizer::Optimizer,
                                  func::MOI.ScalarAffineFunction{Float64},
                                  set::MOI.EqualTo{Float64})
    push!(optimizer.b, MOI.constant(set))
    return AFFEQ(length(optimizer.b))
end

function MOIU.load_constraint(m::Optimizer, ci::AFFEQ,
                              f::MOI.ScalarAffineFunction, s::MOI.EqualTo)
	# See MOIU.load for comments.
    f = MOIU.canonical(f)

	I_s = [Int64[] for i in 1:length(m.blockdims)]
	J_s = [Int64[] for i in 1:length(m.blockdims)]
	V_s = [Float64[] for i in 1:length(m.blockdims)]

    for t in f.terms
        if !iszero(t.coefficient)
            blk, i, j = varmap(m, t.variable_index)
            coef = t.coefficient
            if i != j
                coef /= 2
            end
			push!(I_s[blk], i)
			push!(J_s[blk], j)
			push!(V_s[blk], -coef)
        end
    end

	Ablocks_i = [sparse(I_s[k], J_s[k], V_s[k], abs(m.blockdims[k]), abs(m.blockdims[k])) for k in 1:length(m.blockdims)]
	temp_row = [(m.blockdims[k] > 0) ? sparsevec(svec_lower(Ablocks_i[k])) : sparsevec(diag(Ablocks_i[k])) for k in 1:length(m.blockdims)]
	m.problem.normsA += (norm.(temp_row)).^2
	m.problem.A[:, ci.value] = vcat(temp_row...)
	I_s = nothing
	J_s = nothing
	V_s = nothing
	Ablocks_i = nothing
	temp_row = nothing
end

function MOI.optimize!(m::Optimizer)
    # Update the norms for scaling
	map!(x -> max(1.0, sqrt(sqrt(x))), m.problem.normsA, m.problem.normsA) # NormsA was generated incrementally
	m.problem.normb = max(1.0, norm(m.problem.b))

    # Generate initial iterate uses unscaled data. We scale them later.
	generate_init!(m.problem)
	scale_inner_problem!(m.problem)

    # Set some params.
    m.problem.verbose = !m.silent
	for (key, value) in m.options
		# MOI.set(optimizer, MOI.RawParameter(key), value)
		if key == :maxIter
			m.problem.maxIter = value
		elseif key == :ε_relgap
			m.problem.ε_relgap = value
		elseif key == :ε_φ
			m.problem.ε_φ = value
		elseif key == :verbose
			m.problem.verbose = value
		end
	end
    # Print some info
    if m.problem.verbose == true
		println("======================================")
        println("========    MySDPSolver    ===========")
		println("======================================\n")
    end
	start_time = time()

	solve!(m.problem)

    m.solve_time = time() - start_time
end


function MOI.supports(::Optimizer, ::MOI.TerminationStatus)
	return true
end

function MOI.supports(::Optimizer, ::MOI.PrimalStatus)
	return true
end

function MOI.supports(::Optimizer, ::MOI.DualStatus)
	return true
end

function MOI.get(m::Optimizer, ::MOI.BarrierIterations)
	return m.problem.num_iters
end

function MOI.get(m::Optimizer, ::MOI.TerminationStatus)
	return MOI.OPTIMAL
end

function MOI.get(m::Optimizer, attr::MOI.PrimalStatus)
	return MOI.FEASIBLE_POINT
end
function MOI.get(m::Optimizer, attr::MOI.DualStatus)
	return MOI.FEASIBLE_POINT
end

MOI.get(m::Optimizer, ::MOI.ResultCount) = m.problem === nothing ? 0 : 1

function MOI.get(m::Optimizer, attr::MOI.ObjectiveValue)
	return (-m.objsign)*dot(m.problem.C, m.problem.X) + m.objconstant
    # return m.objsign * dot(m.problem.C, m.problem.X) + m.objconstant
end

function MOI.get(optimizer::Optimizer, attr::MOI.VariablePrimal)

end

function MOI.get(optimizer::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    blk, i, j = varmap(optimizer, vi)
    scaling = (i==j) ? 1.0 : 1.0/sqrt(2.0)
    index = Int(sum([(elem > 0) ? elem*(elem+1)÷2 : abs(elem) for elem in optimizer.blockdims[1:blk-1]])) + 1
    if j >= i
        for ii = 1:abs(optimizer.blockdims[blk])
            for jj = 1:ii
                if (ii, jj) == (i, j)
                    return scaling*optimizer.problem.X[index]
                end
                index += 1
            end
        end
    else
        for ii = 1:abs(optimizer.blockdims[blk])
            for jj = 1:optimizer.blockdims[blk]
                if (jj, ii) == (i, j)
                    return scaling*optimizer.problem.X[index]
                end
                index += 1
            end
        end
    end
end

function MOI.get(optimizer::Optimizer, attr::MOI.ConstraintDual,
                 ci::MOI.ConstraintIndex{MOI.VectorOfVariables, S}) where S<:SupportedSets
	blk, i, j = optimizer.varmap[ci.value]
	scaling = (i==j) ? 1.0 : 1.0/sqrt(2.0)
    start_index = Int(sum([(elem > 0) ? elem*(elem+1)÷2 : abs(elem) for elem in optimizer.blockdims[1:blk-1]])) + 1
	length_block = (optimizer.blockdims[blk] > 0) ? optimizer.blockdims[blk]*(optimizer.blockdims[blk]+1)÷2 : abs(optimizer.blockdims[blk])
	temp = optimizer.problem.S[start_index:start_index+length_block-1]
	idx = 1
	for i = 1:optimizer.blockdims[blk]
		for j = 1:i
			if i != j
				temp[idx] /= sqrt(2.0)
			end
			idx += 1
		end
	end
	return temp
end

function MOI.get(optimizer::Optimizer, attr::MOI.ConstraintPrimal,
	             ci::MOI.ConstraintIndex{MOI.VectorOfVariables, S}) where S<:SupportedSets
	return 1.0
end

function MOI.get(optimizer::Optimizer, attr::MOI.ConstraintDual, ci::AFFEQ)
    # MOI.check_result_index_bounds(optimizer, attr)
    return -optimizer.problem.y[ci.value]
end
