# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

struct KernelDirac{T<:Real} <: AbstractKernelDistribution{T,Discrete}
    value::T
end

Base.show(io::IO, dist::KernelDirac{T}) where {T} = print(io, "KernelDirac{$(T)}, value: $(dist.value)")

Distributions.logpdf(dist::KernelDirac{T}, x) where {T<:Real} = insupport(dist, x) ? zero(T) : typemin(T)

rand_kernel(::AbstractRNG, dist::KernelDirac) = dist.value

Base.maximum(dist::KernelDirac) = dist.value
Base.minimum(dist::KernelDirac) = dist.value
Bijectors.bijector(::KernelDirac) = ZeroIdentity()
Distributions.insupport(dist::KernelDirac, x::Real) = x == dist.value
