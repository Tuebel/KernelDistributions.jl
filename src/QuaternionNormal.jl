# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    QuaternionNormal
Enables to use QuaternionPerturbation as a prior.
"""
struct QuaternionNormal{T} <: AbstractKernelDistribution{Quaternion{T},Continuous}
    μ::Quaternion{T}
    # ndims(::Quaternion) = 0 → do not use vector or BroadcastedDistribution fails
    σ::T
end

QuaternionNormal(::Type{T}=Float32) where {T} = QuaternionNormal{T}(Quaternion(one(T), zero(T), zero(T), zero(T)), one(T))

function Distributions.logpdf(dist::QuaternionNormal{T}, x::Quaternion) where {T}
    w = dist.μ ⊖ nonzero_sign(x)
    sum(logpdf.(KernelNormal.(zero(T), dist.σ), w))
end

rand_kernel(rng::AbstractRNG, dist::QuaternionNormal{T}) where {T} = dist.μ ⊕ rand(rng, KernelNormal(zero(T), dist.σ), 3)

# Bijectors
Bijectors.bijector(::QuaternionNormal) = ZeroIdentity()
# Bijectors.jl only supports x<:Real
Distributions.logpdf(td::UnivariateTransformed{<:QuaternionNormal}, x::Quaternion) = logpdf(td.dist, x)

