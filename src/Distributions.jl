# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# Specifications/Overrides of Distributions.jl

# Manually transform to avoid infinite recursions
rand_kernel(rng::AbstractRNG, transformed_dist::UnivariateTransformed{<:AbstractKernelDistribution}) = transformed_dist.transform(rand(rng, transformed_dist.dist))

# Distributions.jl implementation won't run on the GPU. Only use the most general case, which might be slower but more robust
rand_kernel(rng::AbstractRNG, dist::Truncated{<:AbstractKernelDistribution{T}}) where {T} = invlogcdf(dist.untruncated, logaddexp(T(dist.loglcdf), T(dist.logtp) - randexp(rng, T)))
