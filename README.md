**MySDPSolver** is an interior-point solver meant for solving semidefinite programs. It implements a Mehrotra predictor-corrector algorithm, similar to the one presented in [[1]](#1). It is written entirely in the Julia programming language. It was created as part of my master's thesis. It can solve problems of the following form:  
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\min\limits_{X\in \mathbb{S}^{n}}\:tr(CX)" width=130px>
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\text{s.t.}\: tr(A_{i}X) = b_{i},\:\text{for}\:i=1:m" width=280px>
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=X\succcurlyeq 0" width=50px>
</p>  

## Installation  
* __Requirements__: **MySDPSolver** uses the following packages:
     * [MKLSparse.jl](https://github.com/JuliaSparse/MKLSparse.jl)
## Usage  

Let us solve a simple problem using **MySDPSolver** and [JuMP](https://github.com/JuliaOpt/JuMP.jl).
```julia
# Load packages
using MySDPSolver, JuMP, LinearAlgebra
# Define the data
C = [2.0 1.0
     1.0 2.0]
A = [1.0 0.0
     0.0 0.0]
b = 1.0
# Create the model
model = Model(MySDPSolver.Optimizer)
@variable(model, X[1:2,1:2], SDP)
@objective(model, Min, tr(C*X))
@constraint(model, tr(A*X) == b)
# Call the solving routine
optimize!(model)

```
## References
<a id="1">[1]</a> 
R. H. Tütüncü and K. C. Toh and M. J. Todd (2003).
Solving semidefinite-quadratic-linear programs using SDPT3.
Mathematical Programming, vol.95, 189-217.
