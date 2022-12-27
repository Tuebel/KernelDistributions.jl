# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

struct KernelNormal{T<:Real} <: AbstractKernelDistribution{T,Continuous}
    μ::T
    σ::T
end
KernelNormal(μ::Real, σ::Real) = KernelNormal(promote(μ, σ)...)
KernelNormal(::Type{T}=Float32) where {T} = KernelNormal{T}(0.0, 1.0)

Base.show(io::IO, dist::KernelNormal{T}) where {T} = print(io, "KernelNormal{$(T)}, μ: $(dist.μ), σ: $(dist.σ)")

Distributions.logpdf(dist::KernelNormal, x) = normlogpdf(dist.μ, dist.σ, x)
rand_kernel(rng::AbstractRNG, dist::KernelNormal{T}) where {T} = dist.σ * randn(rng, T) + dist.μ

Base.maximum(::KernelNormal{T}) where {T} = typemax(T)
Base.minimum(::KernelNormal{T}) where {T} = typemin(T)
Bijectors.bijector(::KernelNormal) = ZeroIdentity()
Distributions.insupport(::KernelNormal, ::Real) = true
# Support Truncated{KernelNormal}
Distributions.logcdf(dist::KernelNormal{T}, x::Real) where {T} = normlogcdf(dist.μ, dist.σ, x)
Distributions.invlogcdf(dist::KernelNormal{T}, lp::Real) where {T} = norminvlogcdf(dist.μ, dist.σ, lp)
