# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    ZeroIdentity
Identity bijector without any allocations.
"""
struct ZeroIdentity <: Bijector{0} end
(::ZeroIdentity)(x) = x
Bijectors.inverse(b::ZeroIdentity) = b

Bijectors.logabsdetjac(::ZeroIdentity, x::T) where {T<:Number} = zero(T)
Bijectors.logabsdetjac(::ZeroIdentity, x::AbstractArray{T}) where {T} = zero(T)
Bijectors.logabsdetjac(::Inverse{<:ZeroIdentity}, x::AbstractArray{T}) where {T} = zero(T)
