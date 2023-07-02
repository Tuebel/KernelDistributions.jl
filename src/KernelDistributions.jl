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

using Bijectors
using Base.Broadcast: broadcasted, materialize, Broadcasted
using ChangesOfVariables
using CUDA
using DensityInterface
using LogExpFunctions
using Quaternions
using Random
using Random123: Philox2x, set_counter!
using SpecialFunctions
using StatsFuns

# TODO At one point most of the distributions could be replaced with Distributions.jl. Mixtures could be problematic. A showstopper would be if the discussion around return types would not resolve: https://github.com/JuliaStats/Distributions.jl/issues/1041 . Always returning a Float64 would not be an option for memory constrained GPU calculations.

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

# General
include("Array.jl")
include("Scalar.jl")

# Specialization / Overrides
include("Bijectors.jl")
include("Distributions.jl")

# Univariate
include("BinaryMixture.jl")
include("Dirac.jl")
include("Exponential.jl")
include("Normal.jl")
include("Uniform.jl")

# Special
include("BroadcastedDistribution.jl")
include("TailUniform.jl")
include("CircularUniform.jl")
include("QuaternionUniform.jl")
include("QuaternionPerturbation.jl")
include("SmoothExponential.jl")

export AbstractKernelDistribution

# Array
export array_for_rng
export sum_and_dropdims

# Bijectors
export BroadcastedBijector
export Circular
export ZeroIdentity

# Univariate
export BinaryMixture
export KernelDirac
export KernelExponential
export KernelNormal
export KernelUniform

# Special
export BroadcastedDistribution
export CircularUniform
export QuaternionUniform, ⊕, ⊖
export SmoothExponential
export TailUniform

export param_dims

using Reexport
@reexport import DensityInterface: logdensityof
@reexport import Random: rand!

# Bijectors
@reexport import Bijectors: bijector, inverse, link, invlink, with_logabsdet_jacobian, transformed
@reexport import Distributions: logpdf, logcdf, pdf, truncated


end # module KernelDistributions
