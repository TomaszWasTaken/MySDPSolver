mutable struct inner_problem
	# structure
	block_dims::Vector{Int64}
	m::Int64
	# Data
	A::SparseMatrixCSC{Float64, Int64}
	b::Vector{Float64}
	C::Vector{Float64}

	# Solutions
	X::Vector{Float64}
	y::Vector{Float64}
	S::Vector{Float64}

	# Norms
	normsA::Vector{Float64}
	normb::Float64
	normC::Float64

    num_iters::Int64

	# Params
	verbose::Bool
	maxIter::Int64
	ε_relgap::Float64
	ε_φ::Float64

	function inner_problem(block_dims_::Vector{Int64},
						   A_::SparseMatrixCSC{Float64, Int64},
						   b_::Vector{Float64},
						   c_::Vector{Float64})
		prb = new()
		prb.block_dims = block_dims_
		prb.m = length(b_)
		prb.A = A_
		prb.b = b_
		prb.C = c_
		prb.X = Vector{Float64}(undef, 0)
		prb.y = Vector{Float64}(undef, 0)
		prb.S = Vector{Float64}(undef, 0)
		prb.normsA = Vector{Float64}(undef, 0)
		prb.normb = 1.0
		prb.normC = 1.0
		prb.num_iters = zero(Int64)
		prb.verbose = false
		prb.maxIter = 50
		prb.ε_relgap = 1e-6
		prb.ε_φ = 1e-6
		return prb
	end
end

function scale_inner_problem!(prb::inner_problem)
	nblocks = length(prb.block_dims)
	blk_sizes = [(elem > 0) ? elem*(elem+1)÷2 : abs(elem) for elem in [0; prb.block_dims]]
	blk_indices = cumsum(blk_sizes)
	blocks = [blk_indices[j]+1:blk_indices[j+1] for j = 1:nblocks]
	n = sum(blk_sizes)
	N = sum(abs.(prb.block_dims))

    BLAS.scal!(length(prb.b), 1.0 / prb.normb, prb.b, 1)

	@views for j = 1:nblocks
		BLAS.scal!(blk_sizes[j+1], 1.0 / (prb.normC * prb.normsA[j]) , prb.C[blocks[j]], 1)
		BLAS.scal!(blk_sizes[j+1], prb.normsA[j], prb.X[blocks[j]], 1)
		BLAS.scal!(blk_sizes[j+1], 1.0 / (prb.normC * prb.normsA[j]), prb.S[blocks[j]], 1)
	end

	for j = 1:nblocks
		@. prb.A[blocks[j], :] /= prb.normsA[j]
	end


end

function generate_init!(prb::inner_problem)
	nblocks = length(prb.block_dims)
	ξ = zeros(Float64, nblocks)
	η = zeros(Float64, nblocks)

	temp1 = zeros(Float64, length(prb.b))
	temp2 = zeros(Float64, length(prb.b))

	blk_sizes = [(elem > 0) ? elem*(elem+1)÷2 : abs(elem) for elem in [0; prb.block_dims]] # size of each block, n*(n+1)/2 if nn diag, |n| else
	blk_indices = cumsum(blk_sizes)
	blocks = [blk_indices[j]+1:blk_indices[j+1] for j = 1:nblocks]
	n_elems = sum(blk_sizes)

	@views for j = 1:nblocks
		for k = 1:length(prb.b)
			temp1[k] = (1.0 + abs(prb.b[k])) / (1.0 + norm(prb.A[blocks[j], k]))
			temp2[k] = norm(prb.A[blocks[j], k])
		end
		ξ[j] = abs(prb.block_dims[j]) * maximum(temp1)
		η[j] = (1.0/sqrt(abs(prb.block_dims[j]))) * (1.0 + max(maximum(temp2), norm(prb.C[blocks[j]])))
	end

	X0 = zeros(Float64, n_elems)
	S0 = zeros(Float64, n_elems)

	@views for j = 1:nblocks
		if prb.block_dims[j] > 0
			X0[blocks[j]] = svec(ξ[j]*Matrix{Float64}(I, abs(prb.block_dims[j]), abs(prb.block_dims[j])))
			S0[blocks[j]] = svec(η[j]*Matrix{Float64}(I, abs(prb.block_dims[j]), abs(prb.block_dims[j])))
		else
			X0[blocks[j]] = diag(ξ[j]*Matrix{Float64}(I, abs(prb.block_dims[j]), abs(prb.block_dims[j])))
			S0[blocks[j]] = diag(η[j]*Matrix{Float64}(I, abs(prb.block_dims[j]), abs(prb.block_dims[j])))
		end
	end

	prb.X = X0
	prb.y = zeros(Float64, length(prb.b))
	prb.S = S0
end
