# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    KernelUniform(min, max)
Type stable implementation of a Uniform distribution.
Parameterized by the support [min,max].
If the value is outside the support, logdensityof returns -Inf (alternative [TailUniform](@ref)).
"""
struct KernelUniform{T<:Real} <: AbstractKernelDistribution{T,Continuous}
    min::T
    max::T
end
KernelUniform(min::Real, max::Real) = KernelUniform(promote(min, max)...)
KernelUniform(::Type{T}=Float32) where {T} = KernelUniform{T}(0.0, 1.0)

Base.show(io::IO, dist::KernelUniform{T}) where {T} = print(io, "KernelUniform{$(T)}, a: $(dist.min), b: $(dist.max)")

Distributions.logpdf(dist::KernelUniform{T}, x) where {T<:Real} = insupport(dist, x) ? -log(dist.max - dist.min) : -typemax(T)

rand_kernel(rng::AbstractRNG, dist::KernelUniform{T}) where {T} = (dist.max - dist.min) * rand(rng, T) + dist.min

Base.maximum(dist::KernelUniform) = dist.max
Base.minimum(dist::KernelUniform) = dist.min
# Bijector default for ContinuousUnivariateDistribution is Truncated(minimum, maximum), do not reimplement
Distributions.insupport(dist::KernelUniform, x::Real) = minimum(dist) <= x <= maximum(dist)
Distributions.cdf(d::KernelUniform, x::Real) = clamp((x - d.min) / (d.max - d.min), 0, 1)
