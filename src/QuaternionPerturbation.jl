# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    AdditiveQuaternion(q)
Wraps a quaternion and remaps `+` to `*` and `-` to `/` to support additive proposal interfaces.
"""
struct AdditiveQuaternion{T} <: Number
    q::Quaternion{T}
end

Broadcast.broadcastable(q::AdditiveQuaternion) = Ref(q)
Base.:+(a::Quaternion, b::AdditiveQuaternion) = nonzero_sign(a * b.q)
Base.:-(a::Quaternion, b::AdditiveQuaternion) = nonzero_sign(a / b.q)
Base.abs(q::AdditiveQuaternion) = abs(q.q)

"""
    approx_exponential(x, y, z)
Approximate conversion of small rotation vectors ϕ = (x, y, z) to a quaternion.
About double the speed of the exponential map.
Eq. (193) in J. Sola, „Quaternion kinematics for the error-state KF“
"""
approx_exponential(x::T, y::T, z::T) where {T} = Quaternion(T(1), x / T(2), y / T(2), z / T(2)) |> nonzero_sign

"""
    QuaternionPerturbation
Taylor approximation for small perturbation as described in:
J. Sola, „Quaternion kinematics for the error-state KF“, Laboratoire dAnalyse et dArchitecture des Systemes-Centre national de la recherche scientifique (LAAS-CNRS), Toulouse, France, Tech. Rep, 2012.
"""
struct QuaternionPerturbation{T} <: AbstractKernelDistribution{AdditiveQuaternion{T},Continuous}
    σ_x::T
    σ_y::T
    σ_z::T
end

QuaternionPerturbation(σ=0.01f0::Real) = QuaternionPerturbation(σ, σ, σ)

Distributions.logpdf(dist::QuaternionPerturbation{T}, x::Quaternion) where {T} = normlogpdf(zero(T), dist.σ_x, T(2) * x.v1) + normlogpdf(zero(T), dist.σ_y, T(2) * x.v2) + normlogpdf(zero(T), dist.σ_y, T(2) * x.v3)
Distributions.logpdf(dist::QuaternionPerturbation, x::AdditiveQuaternion) = logpdf(dist, x.q)

rand_kernel(rng::AbstractRNG, dist::QuaternionPerturbation{T}) where {T} = approx_exponential(dist.σ_x * randn(rng, T), dist.σ_y * randn(rng, T), dist.σ_z * randn(rng, T)) |> AdditiveQuaternion

# Bijectors
Bijectors.bijector(::QuaternionPerturbation) = ZeroIdentity()
# Bijectors.jl only supports x<:Real
Distributions.logpdf(td::UnivariateTransformed{<:QuaternionPerturbation}, x::AdditiveQuaternion) = logpdf(td.dist, x)
# Logjact is 0 not Quaternion(0,0,0,0)
Bijectors.logabsdetjac(::ZeroIdentity, x::Union{AdditiveQuaternion{T},AbstractArray{<:AdditiveQuaternion{T}}}) where {T} = zero(T)

