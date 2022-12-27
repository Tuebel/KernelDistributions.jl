# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 


"""
    TailUniform(min, max)
Acts like a uniform distribution of [min,max] but ignores outliers and always returns 1/(max-min) as probability.
"""
struct TailUniform{T<:Real} <: AbstractKernelDistribution{T,Continuous}
    min::T
    max::T
end
TailUniform(min::Real, max::Real) = TailUniform(promote(min, max)...)
TailUniform(::Type{T}=Float32) where {T} = TailUniform{T}(0.0, 1.0)

Base.show(io::IO, dist::TailUniform{T}) where {T} = print(io, "KernelTailUniform{$(T)}, a: $(dist.min), b: $(dist.max)")

Distributions.logpdf(dist::TailUniform{T}, x) where {T<:Real} = -log(dist.max - dist.min)

rand_kernel(rng::AbstractRNG, dist::TailUniform{T}) where {T} = (dist.max - dist.min) * rand(rng, T) + dist.min

Base.maximum(::TailUniform{T}) where {T} = typemax(T)
Base.minimum(::TailUniform{T}) where {T} = typemin(T)
Bijectors.bijector(::TailUniform) = ZeroIdentity()
Distributions.insupport(dist::TailUniform, x::Real) = true
Distributions.cdf(d::TailUniform, x::Real) = clamp((x - d.min) / (d.max - d.min), 0, 1)
