 @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
 Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
 All rights reserved. 

# Basic Distributions
These distributions behave the same way as the ones from [Distributions.jl](https://juliastats.org/Distributions.jl/stable/univariate/).
However, as described in [KernelDistributions.jl](@ref), scalars are always sampled on the CPU even if a `CUDA.RNG` is provided.
They are named accordingly with `Kernel<distribution name>` prefaced.

```@autodocs
Modules = [KernelDistributions]
Pages   = ["Dirac.jl", "Exponential.jl", "Normal.jl", "Uniform.jl"]
```