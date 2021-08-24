**MySDPSolver** is an interior-point solver meant for solving semidefinite programs. It implements a Mehrotra predictor-corrector algorithm, similar to the one presented in [[1]](#1). It is written entirely in the Julia programming language. It was created as part of my master's thesis. It can solve problems of the following form:  
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\min\limits_{X\in \mathbb{S}^{n}}\:tr(CX)" width=130px>
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\text{s.t.}\: tr(A_{i}X) = b_{i},\:\text{for}\:i=1:m" width=300px>
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=X\succcurlyeq 0" width=50px>
</p>  
where  <img src="https://render.githubusercontent.com/render/math?math=X\in \mathbb{S}^{n}"> is the optimization variable and `A_i`, `C`, and `b` are the data of the problem.
## Features

## Installation


## References
<a id="1">[1]</a> 
R. H. Tütüncü and K. C. Toh and M. J. Todd (2003).
Solving semidefinite-quadratic-linear programs using SDPT3.
Mathematical Programming, vol.95, 189-217.
