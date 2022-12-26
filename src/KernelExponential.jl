# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    KernelExponential(β)
Type stable implementation of an (negative) Exponential distribution.
Uses the scale parameter β so the pdf is defined as: inv(β)*exp(inv(β)).
"""
struct KernelExponential{T<:Real} <: AbstractKernelDistribution{T,Continuous}
    β::T
end
KernelExponential(::Type{T}=Float32) where {T} = KernelExponential{T}(1.0)

Base.show(io::IO, dist::KernelExponential{T}) where {T} = print(io, "KernelExponential{$(T)}, θ: $(dist.β)")

function Distributions.logpdf(dist::KernelExponential{T}, x) where {T}
    if insupport(dist, x)
        λ = inv(dist.β)
        -λ * T(x) + log(λ)
    else
        typemin(T)
    end
end

rand_kernel(rng::AbstractRNG, dist::KernelExponential{T}) where {T} = dist.β * randexp(rng, T)

Base.maximum(::KernelExponential{T}) where {T} = typemax(T)
Base.minimum(::KernelExponential{T}) where {T} = zero(T)
Bijectors.bijector(::KernelExponential) = Bijectors.Log{0}()
Distributions.insupport(dist::KernelExponential, x::Real) = minimum(dist) <= x
# Support Truncated{KernelExponential}
Distributions.logcdf(dist::KernelExponential{T}, x::Real) where {T} = log1mexp(-max(T(x) / dist.β, zero(T)))
Distributions.invlogcdf(dist::KernelExponential{T}, lp::Real) where {T} = -log1mexp(T(lp)) * dist.β