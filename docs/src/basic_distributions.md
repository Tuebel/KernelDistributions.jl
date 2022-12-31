# Basic Distributions
These distributions behave the same way as the ones from [Distributions.jl](https://juliastats.org/Distributions.jl/stable/univariate/).
However, as described in [KernelDistributions.jl](@ref), scalars are always sampled on the CPU even if a `CUDA.RNG` is provided.
They are named accordingly with `Kernel<distribution name>` prefaced.
```@index
Pages   = ["basics_distributions.md"]
```
```@docs
BinaryMixture
KernelDirac
KernelExponential
KernelNormal
KernelUniform
```