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

Bijectors.logabsdetjac(::ZeroIdentity, x) = zero(eltype(x))
Bijectors.logabsdetjac(::Inverse{<:ZeroIdentity}, x) = zero(eltype(x))