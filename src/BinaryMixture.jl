# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# Value support makes only sense to be either Discrete or Continuous
"""
    BinaryMixture(dist_1, dist_2, weight_1, weight_2)
Mixture model of two distributions optimized for calculations in the logarithmic domain.
Weights are automatically normalized and transformed to log domain in inner constructor.
"""
struct BinaryMixture{T<:AbstractFloat,S<:ValueSupport,U<:UnivariateDistribution{S},V<:UnivariateDistribution{S}} <: AbstractKernelDistribution{T,S}
    dist_1::U
    dist_2::V
    # Prefer log here, since the logdensity will be used more often than rand
    log_weight_1::T
    log_weight_2::T
    BinaryMixture{T}(dist_1::U, dist_2::V, weight_1, weight_2) where {T,S<:ValueSupport,U<:UnivariateDistribution{S},V<:UnivariateDistribution{S}} = new{T,S,U,V}(dist_1, dist_2, log(weight_1 / (weight_1 + weight_2)), log(weight_2 / (weight_1 + weight_2)))
end

BinaryMixture(dist_1, dist_2, weight_1::T, weight_2::T) where {T} = BinaryMixture{T}(dist_1, dist_2, weight_1, weight_2)
BinaryMixture(dist_1, dist_2, weight_1, weight_2) = BinaryMixture(dist_1, dist_2, promote(weight_1, weight_2)...)

Base.show(io::IO, dist::BinaryMixture{T}) where {T} = print(io, "KernelBinaryMixture{$(T)}\n  components: $(dist.dist_1), $(dist.dist_2) \n  log weights: $(dist.log_weight_1), $(dist.log_weight_2)")

Distributions.logpdf(dist::BinaryMixture{T}, x) where {T} = insupport(dist, x) ? logaddexp(dist.log_weight_1 + logdensityof(dist.dist_1, x), dist.log_weight_2 + logdensityof(dist.dist_2, x)) : -typemax(T)


function rand_kernel(rng::AbstractRNG, dist::BinaryMixture{T}) where {T}
    log_u = log(rand(rng, T))
    if log_u < dist.log_weight_1
        rand(rng, dist.dist_1)
    else
        rand(rng, dist.dist_2)
    end
end

# The support of a mixture is the union of the support of its components
Base.maximum(dist::BinaryMixture) = max(maximum(dist.dist_1), maximum(dist.dist_2))
Base.minimum(dist::BinaryMixture) = min(minimum(dist.dist_1), minimum(dist.dist_2))
Bijectors.bijector(dist::BinaryMixture) = Bijectors.TruncatedBijector(minimum(dist), maximum(dist))
Distributions.insupport(dist::BinaryMixture, x::Real) = minimum(dist) <= x <= maximum(dist)
