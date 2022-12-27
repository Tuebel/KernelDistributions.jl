# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
MeasureTheory.jl is what I have used because of the nicer interface until now, but all the type are not isbits and can not be used on the GPU.
Distributions.jl is pretty close but not perfect for the execution on the GPU:
- Mostly type stable
- Mixtures a quirky
- Uniform is not strongly typed resulting in Float64 calculations all the time.

Here, I provide stripped-down Distributions which are isbitstype, strongly typed and thus support execution on the GPU.
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
Thus, I default to Float32, mostly for memory capacity reasons.
"""
module KernelDistributions
greet() = print("KernelDistributions.jl greetings from Tim.")

using Bijectors
using CUDA
using DensityInterface
using LogExpFunctions
using Random
using Random123: Philox2x, set_counter!
using StatsFuns

# TODO At one point most of the distributions could be replaced with Distributions.jl. Mixtures could be problematic.
# TODO should open a pull request to fix type of https://github.com/JuliaStats/Distributions.jl/blob/d19ac4526bab2584a84323eea4af92805f99f034/src/univariate/continuous/uniform.jl#L120
"""
    AbstractKernelDistribution{T,S<:ValueSupport} <: UnivariateDistribution{S} 
Overrides the following behaviors of Distributions.jl:
- `logdensityof` broadcasts `logpdf`
- `bijector` for an array of distributions broadcasts `bijector`
- Arrays are generated RNG specific (default: Array, CUDA.RNG: CuArray) and filled via broadcasting
"""
abstract type AbstractKernelDistribution{T,S<:ValueSupport} <: UnivariateDistribution{S} end

# WARN parametric alias causes method ambiguities, since the parametric type is always present
const KernelOrTransformedKernel{T} = Union{AbstractKernelDistribution{T},UnivariateTransformed{<:AbstractKernelDistribution{T}},Truncated{<:AbstractKernelDistribution{T}}}

const KernelOrKernelArray = Union{KernelOrTransformedKernel,AbstractArray{<:KernelOrTransformedKernel}}

# TODO does eltype make sense for distributions?
# Base.eltype(::Type{<:KernelOrTransformedKernel{T}}) where {T} = T

include("Array.jl")
include("Scalar.jl")

include("Bijectors.jl")
include("Distributions.jl")

include("KernelExponential.jl")
include("KernelNormal.jl")

export AbstractKernelDistribution
export KernelExponential
export KernelNormal
export ZeroIdentity

using Reexport
@reexport import DensityInterface: logdensityof
@reexport import Random: rand!

# Bijectors
@reexport import Bijectors: bijector, inverse, link, invlink, with_logabsdet_jacobian, transformed
@reexport import Distributions: truncated


end # module KernelDistributions
