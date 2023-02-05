# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# CUDA compatibility
function Bijectors.transform(bij::Bijectors.TruncatedBijector, x::Real)
    a, b = bij.lb, bij.ub
    Bijectors.truncated_link(Bijectors._clamp(x, a, b), a, b)
end

function Bijectors.transform(ib::Inverse{<:Bijectors.TruncatedBijector}, y::Real)
    a, b = ib.orig.lb, ib.orig.ub
    return Bijectors._clamp(Bijectors.truncated_invlink(y, a, b), a, b)
end

function Bijectors.logabsdetjac(b::Bijectors.TruncatedBijector, x::Real)
    a, b = b.lb, b.ub
    return Bijectors.truncated_logabsdetjac(Bijectors._clamp(x, a, b), a, b)
end

# Custom bijectors

"""
    ZeroIdentity
Identity bijector without any allocations.
"""
struct ZeroIdentity <: Bijector end

Bijectors.transform(::ZeroIdentity, x) = x
Bijectors.inverse(b::ZeroIdentity) = b

Bijectors.logabsdetjac(::ZeroIdentity, x::T) where {T<:Number} = zero(T)
Bijectors.logabsdetjac(::ZeroIdentity, x::AbstractArray{T}) where {T} = zero(T)
ChangesOfVariables.with_logabsdet_jacobian(b::ZeroIdentity, x) = b(x), logabsdetjac(b, x)

# Custom reduction like BroadcastedDistribution
"""
    BroadcastedBijector
Uses **lazily** broadcasted to enable bijectors over multiple dimensions.
Moreover reduction dims can be specified which are applied during logabsdetjac correction.
"""
struct BroadcastedBijector{N,B} <: Bijector
    dims::Dims{N}
    bijectors::B
end

"""
    (::BroadcastedBijector)(x)
Applies the internal bijectors via broadcasting.
"""
Bijectors.transform(b::BroadcastedBijector, x) = x .|> b.bijectors

"""
    inverse(b::BroadcastedBijector)
Lazily applies inverse to the internal bijectors.
"""
Bijectors.inverse(b::BroadcastedBijector) = BroadcastedBijector(b.dims, broadcasted(inverse, b.bijectors))

"""
    materialize(b::BroadcastedBijector)
Materialize the possibly broadcasted internal bijectors.
Bijectors are usually required to transform the priors domain, which does not change.
"""
Broadcast.materialize(b::BroadcastedBijector) = BroadcastedBijector(b.dims, materialize(b.bijectors))

"""
    logabsdetjac(b::BroadcastedBijector, x)
Calculate the logabsdetjac correction an reduce the `b.dims` by summing them up.
"""
Bijectors.logabsdetjac(b::BroadcastedBijector, x) = sum_and_dropdims(logabsdetjac.(b.bijectors, x), b.dims)

"""
    with_logabsdet_jacobian(b::BroadcastedBijector, x)
Calculate the transformed variables with the logabsdetjac correction in an optimized fashion.
The logabsdetjac correction is reduced by summing up `b.dims`.
"""
Bijectors.with_logabsdet_jacobian(b::BroadcastedBijector, x) = with_logabsdet_jacobian_array(b, x)
# materialize required for CUDA
Bijectors.with_logabsdet_jacobian(b::BroadcastedBijector{0}, x::AbstractArray) = with_logabsdet_jacobian_array(materialize(b), x)

function with_logabsdet_jacobian_array(b, x)
    with_logjac = with_logabsdet_jacobian.(b.bijectors, x)
    y, logjacs = first.(with_logjac), last.(with_logjac)
    y, sum_and_dropdims(logjacs, b.dims)
end

# Scalar case results in a tuple for with_logjac instead of an array of tuples
Bijectors.with_logabsdet_jacobian(b::BroadcastedBijector{0}, x) = with_logabsdet_jacobian.(b.bijectors, x)
Bijectors.with_logabsdet_jacobian(b::BroadcastedBijector{0}, x::AbstractArray{<:Any,0}) = with_logabsdet_jacobian.(b.bijectors, x)
