[![Run Tests](https://github.com/Tuebel/KernelDistributions.jl/actions/workflows/run_tests.yml/badge.svg)](https://github.com/Tuebel/KernelDistributions.jl/actions/workflows/run_tests.yml)
[![Documenter](https://github.com/Tuebel/KernelDistributions.jl/actions/workflows/documenter.yml/badge.svg)](https://github.com/Tuebel/KernelDistributions.jl/actions/workflows/documenter.yml)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://Tuebel.github.io/KernelDistributions.jl)

# About
This code has been produced during while writing my Ph.D. (Dr.-Ing.) thesis at the institut of automatic control, RWTH Aachen University.
If you find it helpful for your research please cite this:
> T. Redick, „Bayesian inference for CAD-based pose estimation on depth images for robotic manipulation“, RWTH Aachen University, 2024. doi: [10.18154/RWTH-2024-04533](https://doi.org/10.18154/RWTH-2024-04533).

# KernelDistributions.jl
Based on [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) but slimmed down to enable CUDA compatibility.

Distributions are isbitstype, strongly typed and thus support execution on the GPU.
KernelDistributions offer the following interface functions:
- `DensityInterface.logdensityof(dist::KernelDistribution, x)`
- `Random.rand!(rng, dist::KernelDistribution, A)`
- `Base.rand(rng, dist::KernelDistribution, dims...)`
- `Base.eltype(::Type{<:AbstractKernelDistribution})`: Number format of the distribution, e.g. Float16

The Interface requires the following to be implemented:
- Bijectors.bijector(d): Bijector
- `rand_kernel(rng, dist::MyKernelDistribution{T})::T` generate a single random number from the distribution
- `Distributions.logpdf(dist::MyKernelDistribution{T}, x)::T` evaluate the normalized logdensity
- `Base.maximum(d), Base.minimum(d), Distributions.insupport(d)`: Determine the support of the distribution
- `Distributions.logcdf(d, x), Distributions.invlogcdf(d, x)`: Support for Truncated{D}

Most of the time Float64 precision is not required, especially for GPU computations.
Thus, this package defaults to Float32, mostly for memory capacity reasons.
