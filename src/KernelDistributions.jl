module KernelDistributions
greet() = print("KernelDistributions.jl greetings from Tim.")

using Bijectors
using CUDA
using DensityInterface
using LogExpFunctions
using Random
using Random123: Philox2x, set_counter!

# TODO At one point most of the distributions could be replaced with Distributions.jl. Mixtures could be problematic.
# TODO should open a pull request to fix type of https://github.com/JuliaStats/Distributions.jl/blob/d19ac4526bab2584a84323eea4af92805f99f034/src/univariate/continuous/uniform.jl#L120

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

# TODO Distributions only supported for transformed?
abstract type AbstractKernelDistribution{T,S<:ValueSupport} <: UnivariateDistribution{S} end

# WARN parametric alias causes method ambiguities, since the parametric type is always present
const KernelOrTransformedKernel{T} = Union{AbstractKernelDistribution{T},UnivariateTransformed{<:AbstractKernelDistribution{T}},Truncated{<:AbstractKernelDistribution{T}}}

const KernelOrKernelArray = Union{KernelOrTransformedKernel,AbstractArray{<:KernelOrTransformedKernel}}

# TODO does eltype make sense for distributions?
# Base.eltype(::Type{<:KernelOrTransformedKernel{T}}) where {T} = T

include("Array.jl")
include("Scalar.jl")
include("KernelExponential.jl")

include("Distributions.jl")


export KernelExponential

using Reexport
@reexport import DensityInterface: logdensityof
@reexport import Random: rand!

# Bijectors
@reexport import Bijectors: bijector, inverse, link, invlink, with_logabsdet_jacobian, transformed
@reexport import Distributions: truncated


end # module KernelDistributions
