# using LinearAlgebra, SparseArrays

function svec(U)
    # Get the size
    n = size(U, 1)
    N = n*(n+1)÷2
    # Preallocate
    res = zeros(Float64, N)
    index = 1
    # Iterate in column-major order
    for i = 1:n
        for j = 1:i
            if i == j
                res[index] = U[j,i]
            else
                res[index] = sqrt(2.0)*U[j,i]
            end
            index += 1
        end
    end
    return res
end

function svec_lower(U)
    # Get the size
    n = size(U, 1)
    N = n*(n+1)÷2
    # Preallocate
    res = zeros(Float64, N)
    index = 1
    # Iterate in column-major order
    for i = 1:n
        for j = 1:i
            if i == j
                res[index] = U[j,i]
            else
                res[index] = sqrt(2.0)*U[i,j]
            end
            index += 1
        end
    end
    return res
end
function smat(u)
    # Get the size
    N = length(u)
    n = (isqrt(1+8*N)-1)÷2
    # Preallocate
    res = zeros(Float64, n, n)
    index = 1
    for j = 1:n
        for i = 1:j
            if i==j
                res[i,j] = u[index]
            else
                res[i,j] = u[index]/sqrt(2.0)
                res[j,i] = u[index]/sqrt(2.0)
            end
            index += 1
        end
    end
    return res
end


function readSDPA(filename)
    # Open file
    io = open(filename, "r")
    # Read the whole file into a string
    content = read(io, String)
    # Close file
    close(io)
    # Parse the string
    lines = split(content, "\n")
    nlines = length(lines)
    # Number of constraints
    m = parse(Int, lines[1])

    # Number of blocks
    nblocks = parse(Int, lines[2])

    # Number of variables
    blocks = split(lines[3], " ")
    blocksizes = zeros(Int, nblocks)
    blockstarts = zeros(Int, nblocks)
    blockstarts[1] = 1
    n = 0
    for i in 1:nblocks
        n += parse(Int, blocks[i])
        blocksizes[i] = parse(Int, blocks[i])
    end

    for i in 2:nblocks
        blockstarts[i] = blockstarts[i-1] + blocksizes[i]
    end

    # Allocate arrays
    b = zeros(Float64, m)
    bvals = split(lines[4], " ")
    for j in 1:m
        b[j] = -parse(Float64, bvals[j])
    end
    C = zeros(Float64, n, n)
    A = zeros(Float64, n, m*n)

    for l = 5:nlines-1 # \n at the end
        line = split(lines[l], " ")
        cstr_num = parse(Int, line[1])
        block_num = parse(Int, line[2])
        i = parse(Int, line[3])
        j = parse(Int, line[4])
        val = parse(Float64, line[5])
        if cstr_num == 0
            i_index = blockstarts[block_num] + i -1
            j_index = blockstarts[block_num] + j -1
            C[i_index, j_index] = -val
            C[j_index, i_index] = -val
        else
            i_index = blockstarts[block_num] + i -1
            j_index = blockstarts[block_num] + j -1
            A[i_index, (cstr_num-1)*n + j_index] = -val
            A[j_index, (cstr_num-1)*n + i_index] = -val
        end
    end

    return A, b, C
end

function smat!(U, u)
    # Get the size
    n = size(U, 1)
    # Preallocate
    index = 1
    @views for j = 1:n
        for i = 1:j
            if i==j
                U[i,j] = u[index]
            else
                U[i,j] = u[index]/sqrt(2.0)
                U[j,i] = u[index]/sqrt(2.0)
            end
            index += 1
        end
    end
end

function fast_smat!(U, u)
    # Copies a vectorized symmetric matrix u into a full matrix U.

    # Get the size of the matrix and precompute constants.
    n = size(U, 1)
    inv_sqr = 1.0/sqrt(2.0)

    # Computes the index of column in u.
    @inline col_start(col::Int) = (col-1)*col÷2+1

    # Iterate through columns of U.
    @inbounds for k = 1:n
        # Fill the part of column k in the upper triangle of U.
        c_start = col_start(k)
        for j = c_start:c_start+k-2
            U[j-c_start+1, k] = inv_sqr*u[j]
        end
        # Fill the diagonal entry.
        U[k, k] = u[col_start(k)+k-1]
        # Fill the remaining part in the lower triangle.
        for j = k+1:n
            U[j, k] = inv_sqr*u[col_start(j)+k-1]
        end
    end
end
#####################################################################
############## In-place symmetric matrices functions #####################
##########################################################################

function svec!(u, U)

    N = size(U, 1)
    index = 1
    # Iterate in column-major order
    for i = 1:N
        for j = 1:i
            if i == j
                @inbounds u[index] = U[j,i]
            else
                @inbounds u[index] = sqrt(2.0)*U[j,i]
            end
            index += 1
        end
    end
end

function svec_add!(u, U)
    N = size(U, 1)
    n = N*(N+1)÷2
    index = 1
    # Iterate in column-major order
    for i = 1:N
        for j = 1:i
            if i == j
                u[index] += U[j,i]
            else
                u[index] += sqrt(2.0)*U[j,i]
            end
            index += 1
        end
    end
end


function copy_mat!(C, R)
    n = size(R, 1)
    for i = 1:n
        for j = 1:i
            C[j,i] = R[j,i]
        end
    end
    triu!(C)
end

function add_transpose!(U)
    n = size(U, 1)
    for i = 1:n
        for j = 1:n
            @inbounds U[i,j] += U[j,i]
            @inbounds U[i,j] *= 0.5
        end
    end
end

function ind2cart(i)
    m = (isqrt(8*i+1)-1)÷2
    t1 = m*(m+1)÷2
    t2 = (m*m + 3*m +2)÷2
    if t1 == i
        t = t1
    else
        t = max(t1,t2)
    end
    y = isqrt(2*t)
    x = y - (t-i)
    # println(x,", ",y)
    return x, y
end

function sparse_A_mul(C, a, B, colptr, rowval, nzval)
    n = size(B, 1)
    NNZ = length(a.nzind)

    fill!(colptr, zero(BlasInt))
    fill!(rowval, zero(BlasInt))
    fill!(nzval, zero(Float64))

    for k = 1:length(a.nzind)
        i, j = ind2cart(a.nzind[k])
        colptr[j+1] += 1
        rowval[k] = i
        nzval[k] = i == j ? a.nzval[k] : a.nzval[k]/sqrt(2.0)
    end

    colptr[1] = 1
    for k = 1:n
        colptr[k+1] += colptr[k]
    end

    mat = SparseMatrixCSC{Float64, BlasInt}(n, n, colptr, rowval, nzval)
    # display(colptr)
    MKLSparse.BLAS.cscmm!('N', 1.0, "SUNF", mat, B, 0.0, C)
end
