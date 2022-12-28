# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    robust_normalize(q)
Compared to the implementation in Quaternions.jl, this implementation takes care of the re-normalization and avoiding divisions by zero.
Eq. (44) in J. Sola, „Quaternion kinematics for the error-state KF“.
"""
function robust_normalize(q::Quaternion{T}) where {T}
    a = abs(q)
    if iszero(a)
        Quaternion(one(T), zero(T), zero(T), zero(T), true)
    else
        # Rotations.jl likes normalized quaternions → do not ignore small deviations from 1
        qn = q / a
        Quaternion(qn.s, qn.v1, qn.v2, qn.v3, true)
    end
end

"""
    QuaternionUniform
Allows true uniform sampling of 3D rotations.
Normalization requires scalar indexing, thus CUDA is not supported.
"""
struct QuaternionUniform{T} <: AbstractKernelDistribution{Quaternion{T},Continuous} end

QuaternionUniform(::Type{T}=Float32) where {T} = QuaternionUniform{T}()

# Quaternions lie on the surface of a 4D hypersphere with radius 1. Due to to the duality of quaternions, it is only the surface of the half sphere.
# https://marc-b-reynolds.github.io/quaternions/2017/11/10/AveRandomRot.html 
const quat_logp = -log(π^2)
Distributions.logpdf(::QuaternionUniform{T}, x::Quaternion) where {T} = T(quat_logp)

rand_kernel(rng::AbstractRNG, ::QuaternionUniform{T}) where {T} = Quaternion(randn(rng, T), randn(rng, T), randn(rng, T), randn(rng, T)) |> robust_normalize

# Bijectors
Bijectors.bijector(::QuaternionUniform) = ZeroIdentity()
# Logjact is 0 not Quaternion(0,0,0,0)
Bijectors.logabsdetjac(::ZeroIdentity, x::Union{Quaternion{T},AbstractArray{<:Quaternion{T}}}) where {T} = zero(T)
Bijectors.logabsdetjac(::Inverse{<:ZeroIdentity}, x::Union{Quaternion{T},AbstractArray{<:Quaternion{T}}}) where {T} = zero(T)
# Bijectors.jl only supports x<:Real
Distributions.logpdf(td::UnivariateTransformed{<:QuaternionUniform}, x::Quaternion) = logpdf(td.dist, x)
