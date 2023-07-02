# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    nonzero_sign(q)
Compared to the implementation in Quaternions.jl, this implementation does not return a Quaternion with (0,0,0,0) but the identity with (1,0,0,0).
Eq. (44) in J. Sola, „Quaternion kinematics for the error-state KF“.
"""
function nonzero_sign(q::Quaternion{T}) where {T}
    normalized = sign(q)
    if iszero(normalized)
        Quaternion(one(T), zero(T), zero(T), zero(T))
    else
        normalized
    end
end

"""
    QuaternionUniform
Allows true uniform sampling of 3D rotations on CPU & GPU (CUDA).
"""
struct QuaternionUniform{T} <: AbstractKernelDistribution{Quaternion{T},Continuous} end

QuaternionUniform(::Type{T}=Float32) where {T} = QuaternionUniform{T}()

# Quaternions lie on the surface of a 4D hypersphere with radius 1. Due to to the duality of quaternions, it is only the surface of the half sphere.
# https://marc-b-reynolds.github.io/quaternions/2017/11/10/AveRandomRot.html 
const quat_logp = -log(π^2)
Distributions.logpdf(::QuaternionUniform{T}, x::Quaternion) where {T} = T(quat_logp)

rand_kernel(rng::AbstractRNG, ::QuaternionUniform{T}) where {T} = randn(rng, Quaternion{T}) |> nonzero_sign

# Bijectors
Bijectors.bijector(::QuaternionUniform) = ZeroIdentity()
# Bijectors.jl only supports x<:Real
Distributions.logpdf(td::UnivariateTransformed{<:QuaternionUniform}, x::Quaternion) = logpdf(td.dist, x)
# Logjact is 0 not Quaternion(0,0,0,0)
Bijectors.logabsdetjac(::ZeroIdentity, x::Union{Quaternion{T},AbstractArray{<:Quaternion{T}}}) where {T} = zero(T)

