# include("utils.jl")
# include("problem.jl")

function solve!(prb::inner_problem)
    # Set up BLAS
    BLAS.set_num_threads(4)

    # prb.X = prb.X
    # prb.y = prb.y
    # prb.S = prb.S

    #=
    Compute the corresponding linear indices for each block.
    =#
    nblocks = length(prb.block_dims)
    blk_sizes = [(elem > 0) ? elem*(elem+1)÷2 : abs(elem) for elem in [0; prb.block_dims]]  # |n| if diagonal, n(n+1)÷2 else
    blk_indices = cumsum(blk_sizes)
    blocks = [blk_indices[j]+1:blk_indices[j+1] for j = 1:nblocks]                          # linear indices for each block
    n = sum(blk_sizes)
    N = sum(abs.(prb.block_dims))
    ns = [0; prb.block_dims]

    #=
    Allocate the residuals and compute initial values.
    =#
    rp = Vector{Float64}(undef, prb.m)
    Rd = Vector{Float64}(undef, n)
    Rc = Vector{Float64}(undef, n)

    BLAS.blascopy!(prb.m, prb.b, 1, rp, 1)  # b ← rp
    BLAS.blascopy!(n, prb.C, 1, Rd, 1)      # svec(C) ← Rd
    BLAS.axpy!(-1.0, prb.S, Rd)                # Rd - prb.S ← Rd

    MKLSparse.BLAS.cscmv!('T', -1.0, "GU2F", prb.A, prb.X, 1.0, rp)  # b - 𝓐prb.X ← rp
    MKLSparse.BLAS.cscmv!('N', -1.0, "GU2F", prb.A, prb.y, 1.0, Rd)  # Rd - prb.S - 𝓐ᵀprb.y ← Rd

    #=
    Compute μ and other statistics.
    =#
    μ = BLAS.dot(prb.X, prb.S) / N
    relgap = BLAS.dot(prb.X, prb.S) / (1.0 + max(abs(BLAS.dot(prb.C, prb.X)), abs(BLAS.dot(prb.b, prb.y))))
    pinfeas = norm(rp) / (1.0 + norm(prb.b))
    dinfeas = norm(Rd) / (1.0 + norm(prb.C))
    φ = max(pinfeas, dinfeas)
    solution_relgap = prb.ε_relgap  # default value: 1e-6
    solution_φ = prb.ε_φ            # default value: 1e-6
    g = 0.9                         # if adaptative
    # g = 0.98                        # if constant

    sdp_blocks_indices = zeros(Int64, nblocks)
    n_sdp_blocks = 0

    for j = 1:nblocks
        if ns[j+1] > 0
            sdp_blocks_indices[j] = n_sdp_blocks+1
            n_sdp_blocks += 1
        end
    end

    #=
    Preallocate additional memory to perform in-place matrix operations for semidefinite blocks.
    Using Vectors of Matrices is more julianic than using slices.
    =#
    L = [Matrix{Float64}(undef, abs(ns[j+1]), abs(ns[j+1])) for j in 1:nblocks if sdp_blocks_indices[j] > 0]
    R = [Matrix{Float64}(undef, abs(ns[j+1]), abs(ns[j+1])) for j in 1:nblocks if sdp_blocks_indices[j] > 0]
    U = [Matrix{Float64}(undef, abs(ns[j+1]), abs(ns[j+1])) for j in 1:nblocks if sdp_blocks_indices[j] > 0]
    U2 = [Matrix{Float64}(undef, abs(ns[j+1]), abs(ns[j+1])) for j in 1:nblocks if sdp_blocks_indices[j] > 0]
    U3 = [Matrix{Float64}(undef, abs(ns[j+1]), abs(ns[j+1])) for j in 1:nblocks if sdp_blocks_indices[j] > 0]

    # MAX_SDP_BLOCK_SIZE = maximum(abs(ns[j+1]) for j in 1:nblocks if sdp_blocks_indices[j] > 0)
    # SDP_SIZE = [ns[j+1] for j in 1:nblocks if sdp_blocks_indices[j] > 0]
    # U_ = Matrix{Float64}(undef, MAX_SDP_BLOCK_SIZE, MAX_SDP_BLOCK_SIZE)
    # U2_ = Matrix{Float64}(undef, MAX_SDP_BLOCK_SIZE, MAX_SDP_BLOCK_SIZE)
    # U3_ = Matrix{Float64}(undef, MAX_SDP_BLOCK_SIZE, MAX_SDP_BLOCK_SIZE)

    m_indices = [Int64[] for j = 1:nblocks if sdp_blocks_indices[j] > 0]
    for j = 1:nblocks
        if sdp_blocks_indices[j] > 0
            index = sdp_blocks_indices[j]
            for k = 1:prb.m
                if nnz(prb.A[blocks[j], k]) > 0
                    push!(m_indices[index], k)
                end
            end
        end
    end

    #=
    Additional memory to solve for Δy.
    =#
    BB = zeros(Float64, size(prb.A))
    B = zeros(Float64, size(prb.A, 2), size(BB, 2))
    h = Vector{Float64}(undef, prb.m)
    hh = Matrix{Float64}(undef, prb.m, 1)

    H1 = zeros(Float64, size(Rd))
    H2 = zeros(Float64, size(prb.X))

    #=
    Steps
    =#
    δX = Vector{Float64}(undef, size(prb.X))
    δS = Vector{Float64}(undef, size(prb.S))
    δy = zeros(Float64, prb.m)

    αs = zeros(Float64, length(ns)-1)  # Computation of max. step-length
    βs = zeros(Float64, length(ns)-1)

    τ = Vector{Float64}(undef, prb.m)  # Stores the Householder vectors
    iter = 1
    iterMax = prb.maxIter

    #=
    Extra-memory for sparse devectorized matrices Aᵢ.
    =#

    # We take the matrix Aᵢ with most non-zero elements.
    # NNZ_MAX = maximum(nnz(prb.A[:, col]) for col = 1:size(prb.A,2))
    NNZ_MAX = maximum(nnz(prb.A[blocks[j], col]) for col = 1:size(prb.A, 2) for j = 1:nblocks if sdp_blocks_indices[j] > 0)
    # println("new, old: ", NNZ_MAX, ", ", NNZ_MAX_prev)
    # We take the biggest semidefinite block.
    N_MAX = (n_sdp_blocks > 0) ? maximum(abs(ns[j+1]) for j in 1:nblocks if sdp_blocks_indices[j] > 0) : 1
    # The vectors needed for a CSC sparse matrix.
    col_cache = Vector{BlasInt}(undef, N_MAX+1)
    row_cache = Vector{BlasInt}(undef, NNZ_MAX)
    nnz_cache = Vector{Float64}(undef, NNZ_MAX)

    # Print the header if verbose
    if prb.verbose == true
        @printf("iter   |   pobj (scal.)  |   dobj (scal.)\n")
    end

    #=
    Main loop
    =#
    @inbounds while (relgap > solution_relgap) || (φ > solution_φ)
        @views @inbounds for j = 1:nblocks
            if sdp_blocks_indices[j] > 0
                index = sdp_blocks_indices[j]

                fast_smat!(L[index], prb.X[blocks[j]])  # Devectorize the matrices.
                fast_smat!(R[index], prb.S[blocks[j]])

                # BLAS.symm!('L','U', -1.0, L[index], R[index], 0.0, U2[index])
                LAPACK.potrf!('U', L[index])         # Compute Cholesky factorizations.
                LAPACK.potrf!('U', R[index])

                # copyto!(U5[index], 1.0I)             # R\I ← U5
                # LAPACK.trtrs!('U', 'N', 'N', R[index], U5[index])

                # triu!(L[index])
                triu!(R[index])

                # BLAS.gemm!('N','T', 1.0, R[index], L[index], 0.0, U[index])
                # copyto!(U[index], R[index])
                BLAS.blascopy!(length(U[index]), R[index], 1, U[index], 1)
                BLAS.trmm!('R','U','T','N', 1.0, L[index], U[index])
                BLAS.syrk!('U','N', -1.0, U[index], 0.0, U2[index])

                svec!(Rc[blocks[j]], U2[index])
            # @time begin
                # BLAS.blascopy!(SDP_SIZE[index]^2, R[index], 1, U_[1:SDP_SIZE[index],1:SDP_SIZE[index]], 1)
                # BLAS.trmm!('R','U','T','N', 1.0, L[index], U_[1:SDP_SIZE[index],1:SDP_SIZE[index]])
                # BLAS.syrk!('U','N', -1.0, U_[1:SDP_SIZE[index],1:SDP_SIZE[index]], 0.0, U2_[1:SDP_SIZE[index],1:SDP_SIZE[index]])
                #
                # svec!(Rc[blocks[j]], U2_[1:SDP_SIZE[index],1:SDP_SIZE[index]])
            # end
            else
                for i in blocks[j]
                    Rc[i] = -prb.X[i]*prb.S[i]
                end
            end
        end

        #=
        Update the residuals.
        =#
        BLAS.blascopy!(prb.m, prb.b, 1, rp, 1)
        BLAS.blascopy!(n, prb.C, 1, Rd, 1)
        BLAS.axpy!(-1.0, prb.S, Rd)

        MKLSparse.BLAS.cscmv!('T', -1.0, "GU2F", prb.A, prb.X, 1.0, rp)
        MKLSparse.BLAS.cscmv!('N', -1.0, "GU2F", prb.A, prb.y, 1.0, Rd)

        μ = BLAS.dot(prb.X, prb.S) / N
        relgap = BLAS.dot(prb.X, prb.S) / (1.0 + max(abs(BLAS.dot(prb.C, prb.X)), abs(BLAS.dot(prb.b, prb.y))))
        pinfeas = norm(rp) / (1.0 + norm(prb.b))
        dinfeas = norm(Rd) / (1.0 + norm(prb.C))
        φ = max(pinfeas, dinfeas)

        for j = 1:nblocks
            if sdp_blocks_indices[j] > 0
                index = sdp_blocks_indices[j]

                @views fast_smat!(U[index], prb.X[blocks[j]])

                # @views copyto!(U2[index], 1.0I)
                # @views BLAS.trsm!('L','U','T','N', 1.0, R[index], U2[index])
                # @views BLAS.trsm!('L','U','N','N', 1.0, R[index], U2[index])

                # for k = 1:prb.m
                for k in m_indices[index]
                    # if iter== 3 && all(prb.A[blocks[j], k] .== 0.0)
                    #     count_zero += 1
                    # end
                    sparse_A_mul(U2[index], prb.A[blocks[j], k], U[index], col_cache, row_cache, nnz_cache)
                    # @views BLAS.symm!('L','U', 1.0, U2[index], U4[index], 0.0, U3[index])
                    @views BLAS.trsm!('L','U','T','N', 1.0, R[index], U2[index])
                    @views BLAS.trsm!('L','U','N','N', 1.0, R[index], U2[index])
                    @views add_transpose!(U2[index])
                    @views svec!(BB[blocks[j], k], U2[index])
                end

                @views fast_smat!(U2[index], Rd[blocks[j]])
                @views BLAS.symm!('R','U', 1.0, U[index], U2[index], 0.0, U3[index])
                # @views BLAS.symm!('L','U', 1.0, U2[index], U4[index], 0.0, U3[index])
                @views BLAS.trsm!('L','U','T','N', 1.0, R[index], U3[index])
                @views BLAS.trsm!('L','U','N','N', 1.0, R[index], U3[index])
                @views add_transpose!(U3[index])
                @views svec!(H1[blocks[j]], U3[index])

                @views fast_smat!(U2[index], Rc[blocks[j]])
                # @views BLAS.trmm!('R','U','T','N', 1.0, U5[index], U3[index])
                # @views BLAS.trmm!('L','U','N','N', 1.0, U5[index], U3[index])
                @views BLAS.trsm!('L','U','N','N', 1.0, R[index], U2[index])
                @views BLAS.trsm!('R','U','T','N', 1.0, R[index], U2[index])
                @views add_transpose!(U2[index])
                @views svec!(H2[blocks[j]], U2[index])
            else
                @views for i in blocks[j]
                    for k = 1:prb.m
                        BB[i, k] = prb.A[i,k]*prb.X[i]/prb.S[i]
                    end
                    H1[i] = Rd[i]*prb.X[i]/prb.S[i]
                    H2[i] = Rc[i]/prb.S[i]
                end
            end
        end

        MKLSparse.BLAS.cscmm!('T', 1.0, "GU2F", prb.A, BB, 0.0, B)      # 𝓐BB ← B
        BLAS.blascopy!(prb.m, rp, 1, h, 1)
        MKLSparse.BLAS.cscmv!('T', 1.0, "GU2F", prb.A, H1-H2, 1.0, h)   # h

        #=
        Solve the linear system for δy (Predictor step).
        =#
        BLAS.blascopy!(prb.m, h, 1, hh, 1)
        LAPACK.geqrf!(B, τ)                         # QR factorization.
        LAPACK.ormqr!('L','T', B, τ, hh)            # Apply factors to the RHS.
        BLAS.trsm!('L','U','N','N', 1.0, B, hh)     # Triangular solve.
        BLAS.blascopy!(prb.m, hh, 1, δy, 1)         # Copy solution.

        # Compute δS
        BLAS.blascopy!(n, Rd, 1, δS, 1)
        MKLSparse.BLAS.cscmv!('N', -1.0, "GU2F", prb.A, δy, 1.0, δS)

        # Compute δX
        @views for j = 1:nblocks
            if sdp_blocks_indices[j] > 0
                index = sdp_blocks_indices[j]
                fast_smat!(U2[index], δS[blocks[j]]) # S\I * δS * X
                BLAS.symm!('L','U', 1.0, U2[index], U[index], 0.0, U3[index])
                # BLAS.symm!('L','U', 1.0, U2[index], U4[index], 0.0, U3[index])
                BLAS.trsm!('L','U','T','N', 1.0, R[index], U3[index])
                BLAS.trsm!('L','U','N','N', 1.0, R[index], U3[index])
                add_transpose!(U3[index])
                svec!(δX[blocks[j]], U3[index])
            else
                for i in blocks[j]
                    δX[i] = δS[i]*prb.X[i]/prb.S[i]
                end
            end
        end

        BLAS.axpby!(-1.0, prb.X, -1.0, δX)

        #=
        Computation of the minimum eigenvalues.
        =#
        @views for j = 1:length(ns)-1
            if sdp_blocks_indices[j] > 0
                index = sdp_blocks_indices[j]
                fast_smat!(U[index], δX[blocks[j]])

                BLAS.trsm!('R','U','N','N', 1.0, L[index], U[index])
                BLAS.trsm!('L','U','T','N', 1.0, L[index], U[index])
                λ_a = LAPACK.syevr!('N','I','U', U[index], 0.0, 0.0, 1, 1, 1e-6)[1][1]

                fast_smat!(U[index], δS[blocks[j]])
                BLAS.trsm!('R','U','N','N', 1.0, R[index], U[index])
                BLAS.trsm!('L','U','T','N', 1.0, R[index], U[index])
                λ_b = LAPACK.syevr!('N','I','U', U[index], 0.0, 0.0, 1, 1, 1e-6)[1][1]
            else
                λ_a = minimum(δX[blocks[j]] ./ prb.X[blocks[j]])
                λ_b = minimum(δS[blocks[j]] ./ prb.S[blocks[j]])
            end
            αs[j] = (λ_a < 0.0) ? -1.0/λ_a : Inf
            βs[j] = (λ_b < 0.0) ? -1.0/λ_b : Inf
        end

        α = min(1.0, g*minimum(αs))     # max. step lengths
        β = min(1.0, g*minimum(βs))

        #=
        Update of σ.
        =#
        if μ > 1e-6
            if min(α,β) < 1.0/sqrt(3.0)
                exp = 1
            else
                exp = max(1.0, 3*min(α,β)^2)
            end
        else
            exp = 1
        end

        if dot(prb.X+α*δX, prb.S+β*δS) < 0.0
            σ = 0.8
        else
            frac = dot(prb.X+α*δX, prb.S+β*δS)/dot(prb.X, prb.S)
            σ = min(1.0, frac^exp)
        end

        #=
        Update of the residual Rc for the corrector step.
        =#
        @views for j = 1:nblocks
            if sdp_blocks_indices[j] > 0
                index = sdp_blocks_indices[j]
                fast_smat!(U[index], δX[blocks[j]])
                fast_smat!(U2[index], δS[blocks[j]])
                BLAS.symm!('L','U', 1.0, U[index], U2[index], 0.0, U3[index])
                BLAS.trmm!('L','U','N','N', 1.0, R[index], U3[index])
                BLAS.trsm!('R','U','N','N', 1.0, R[index], U3[index])
                add_transpose!(U3[index])
                svec_add!(Rc[blocks[j]], μ*σ*I - U3[index])
            else
                for i in blocks[j]
                    Rc[i] += μ*σ - δX[i]*δS[i]
                end
            end
        end

        @views for j = 1:nblocks
            if sdp_blocks_indices[j] > 0
                index = sdp_blocks_indices[j]
                fast_smat!(U[index], prb.X[blocks[j]])
                fast_smat!(U2[index], Rd[blocks[j]])

                BLAS.symm!('L','U', 1.0, U2[index], U[index], 0.0, U3[index])
                # BLAS.symm!('L','U', 1.0, U3[index], U4[index], 0.0, U2[index])
                BLAS.trsm!('L','U','T','N', 1.0, R[index], U3[index])
                BLAS.trsm!('L','U','N','N', 1.0, R[index], U3[index])

                add_transpose!(U3[index])
                svec!(H1[blocks[j]], U3[index])

                fast_smat!(U3[index], Rc[blocks[j]])
                # BLAS.trmm!('R','U','T','N', 1.0, U5[index], U4[index])
                # BLAS.trmm!('L','U','N','N', 1.0, U5[index], U4[index])
                BLAS.trsm!('L','U','N','N', 1.0, R[index], U3[index])
                BLAS.trsm!('R','U','T','N', 1.0, R[index], U3[index])

                add_transpose!(U3[index])
                svec!(H2[blocks[j]], U3[index])
            else
                for i in blocks[j]
                    H1[i] = Rd[i]*prb.X[i]/prb.S[i]
                    H2[i] = Rc[i]/prb.S[i]
                end
            end
        end

        BLAS.blascopy!(prb.m, rp, 1, h, 1)
        MKLSparse.BLAS.cscmv!('T', 1.0, "GU2F", prb.A, H1-H2, 1.0, h)

        #=
        Solving the linear system for Δy (Corrector step).
        =#
        BLAS.blascopy!(prb.m, h, 1, hh, 1)
        LAPACK.ormqr!('L','T', B, τ, hh)
        BLAS.trsm!('L','U','N','N', 1.0, B, hh)
        BLAS.blascopy!(prb.m, hh, 1, δy, 1)

        # ΔS
        BLAS.blascopy!(n, Rd, 1, δS, 1)
        MKLSparse.BLAS.cscmv!('N', -1.0, "GU2F", prb.A, δy, 1.0, δS)

        # ΔX
        @views @inbounds for j = 1:nblocks
            if sdp_blocks_indices[j] > 0
                index = sdp_blocks_indices[j]
                fast_smat!(U2[index], δS[blocks[j]])

                # copyto!(U3[index], 1.0I)
                # BLAS.trsm!('L','U','T','N', 1.0, R[index], U3[index])
                # BLAS.trsm!('L','U','N','N', 1.0, R[index], U3[index])
                #
                # BLAS.symm!('L','U', 1.0, U2[index], U[index], 0.0, U4[index])
                # BLAS.symm!('L','U', 1.0, U3[index], U4[index], 0.0, U2[index])
                BLAS.trsm!('L','U','T','N', 1.0, R[index], U2[index])
                BLAS.trsm!('L','U','N','N', 1.0, R[index], U2[index])
                BLAS.symm!('R','U', 1.0, U[index], U2[index], 0.0, U3[index])

                add_transpose!(U3[index])
                svec!(H1[blocks[j]], U3[index])
            else
                for i in blocks[j]
                    H1[i] = δS[i]*prb.X[i]/prb.S[i]
                end
            end
        end

        δX .= H2 - H1

        #=
        Minimum eigenvalues for the max. step-length.
        =#
        @views for j = 1:length(ns)-1
            if sdp_blocks_indices[j] > 0
                index = sdp_blocks_indices[j]

                fast_smat!(U[index], δX[blocks[j]])
                BLAS.trsm!('R','U','N','N', 1.0, L[index], U[index])
                BLAS.trsm!('L','U','T','N', 1.0, L[index], U[index])
                λ_a = LAPACK.syev!('N','U', U[index])[1]

                fast_smat!(U[index], δS[blocks[j]])
                BLAS.trsm!('R','U','N','N', 1.0, R[index], U[index])
                BLAS.trsm!('L','U','T','N', 1.0, R[index], U[index])
                λ_b = LAPACK.syev!('N','U', U[index])[1]

            else
                λ_a = minimum(δX[blocks[j]] ./ prb.X[blocks[j]])
                λ_b = minimum(δS[blocks[j]] ./ prb.S[blocks[j]])
            end
            αs[j] = (λ_a < 0.0) ? -1.0/λ_a : Inf
            βs[j] = (λ_b < 0.0) ? -1.0/λ_b : Inf
        end

        α = min(1.0, g*minimum(αs))
        β = min(1.0, g*minimum(βs))

        g = 0.9 + 0.09*min(α, β)  # adaptative update

        BLAS.axpy!(α, δX, prb.X)
        BLAS.axpy!(β, δS, prb.S)
        BLAS.axpy!(β, δy, prb.y)

        pinfeas = norm(rp) / (1.0 + norm(prb.b))
        dinfeas = norm(Rd) / (1.0 + norm(prb.C))

        # Print iteration info if verbose
        if prb.verbose == true
            @printf("%2d     |  %.3e     |  %.3e\n", iter, BLAS.dot(prb.C, prb.X), BLAS.dot(prb.b, prb.y))
        end

        iter += 1
        if iter == iterMax
            break
        end

    end

    # Print the last iteration info
    if prb.verbose == true
        @printf("%2d     |  %.3e     |  %.3e\n", iter, BLAS.dot(prb.C, prb.X), BLAS.dot(prb.b, prb.y))
    end

    # Infeasibility tests
    inftol = 1e-8
    if BLAS.dot(prb.b, prb.y) / norm(prb.A*prb.y + prb.S) > 1/inftol
        println("Primal likely infeasible")
    end
    if -BLAS.dot(prb.C, prb.X) / norm(prb.A'*prb.X) > 1/inftol
        println("Dual likely infeasible")
    end

    prb.num_iters = iter

    @views for j = 1:length(ns)-1
        prb.X[blocks[j]] *= (prb.normb / prb.normsA[j])
        prb.S[blocks[j]] *= (prb.normC * prb.normsA[j])
    end

    prb.y *= prb.normC

    # @. prb.X = prb.X
    # @. prb.y = prb.y
    # @. prb.S = prb.S
    lmul!(prb.normb, prb.b)

    # Unscaling
    @views for j = 1:length(ns)-1
        prb.A[blocks[j],:] *= prb.normsA[j]
        prb.C[blocks[j]] *= prb.normC*prb.normsA[j]
    end

    if prb.verbose == true
        println("")
        println("Primal value: ", BLAS.dot(prb.C, prb.X))
        println("Dual value: ", BLAS.dot(prb.b, prb.y))
        println("N iters: ", iter)
    end
end
