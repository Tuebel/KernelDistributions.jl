# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    BroadcastedDistribution{T,N,M}
A **lazy** implementation for multi-dimensional distributions which makes it natural to use on different devices and applying transformations afterwards.

At the core, `marginals` is a broadcasted distribution for a set of parameters.
Generating random numbers is based on the promoted type `T` of the parameters and stored in.
Logdensities are evaluated using broadcasted and reduced by summing up `dims`, similar to a product distribution.
The reduction dimensions `N` might differ from the dimensions of the parameters, in case that the parameters represent multiple samples.
"""
struct BroadcastedDistribution{T,N,M<:Broadcasted}
    dims::Dims{N}
    marginals::M
end
const ScalarBroadcastedDistribution = BroadcastedDistribution{<:Any,<:Any,<:Broadcasted{<:Broadcast.DefaultArrayStyle{0}}}

BroadcastedDistribution(::Type{T}, dims::Dims{N}, marginals::M) where {T,N,M<:Broadcasted} = BroadcastedDistribution{T,N,M}(dims, marginals)

"""
    BroadcastedDistribution(dist, dims, params...)
Construct a BroadcastedDistribution for a distribution generating function, conditioned on params.
The `dims` of the distribution which are reduced are set manually so they can differ from the dims of the parameters.
"""
BroadcastedDistribution(dist, dims::Dims, params...) =
    BroadcastedDistribution(promote_params_eltype(params...), dims, broadcasted(dist, params...))

"""
    BroadcastedDistribution(dist, params...)
Construct a BroadcastedDistribution for a distribution generating function, conditioned on params.
Automatically reduces all dimensions of the parameters, like a product distribution.
"""
BroadcastedDistribution(dist, params...) = BroadcastedDistribution(promote_params_eltype(params...), param_dims(params...), broadcasted(dist, params...))

"""
    promote_params_eltype(params...)
Promote the types of the elements in params to get the minimal common type.
"""
promote_params_eltype(params...) = promote_type(eltype.(params)...)

"""
    n_param_dims(params...)
Finds the maximum ndims of the parameters.
"""
n_param_dims(params...) = maximum(ndims.(params))

"""
    param_dims(params...)
Finds the maximum possible Dims of the parameters.
"""
param_dims(params...) = (1:n_param_dims(params...)...,)

"""
    marginals(dist)
Lazy broadcasted array of distributions → use dot syntax, Broadcast.broadcasted([...], marginals) or Broadcast.materialize(marginals).
"""
marginals(dist::BroadcastedDistribution) = dist.marginals

Base.show(io::IO, dist::BroadcastedDistribution{T}) where {T} = print(io, "BroadcastedDistribution{$(T)}\n  dist function: $(recursive_marginals_string(marginals(dist)))\n  size: $(size(dist))\n  dims: $(dist.dims)\n")

"""
    recursive_marginals_string
Recursively generates a string of the distribution type (function) of the broadcasted marginals.
"""
function recursive_marginals_string(marginals)
    res = "$(marginals.f) "
    if isempty(marginals.args)
        res
    elseif marginals.args[1] isa Broadcasted
        res *= recursive_marginals_string(marginals.args[1])
    end
    res
end

Base.axes(dist::BroadcastedDistribution) = axes(dist.marginals)
Base.Dims(dist::BroadcastedDistribution) = dist.dims
# Might differ from the dims of the marginals
Base.ndims(::BroadcastedDistribution{<:Any,N}) where {N} = N
Base.size(dist::BroadcastedDistribution) = size(dist.marginals)


# DensityInterface

DensityInterface.logdensityof(dist::BroadcastedDistribution, x) = logpdf(dist, x)
@inline DensityInterface.DensityKind(::BroadcastedDistribution) = HasDensity()

"""
    logpdf(dist, x)
Evaluate the logdensity of multi-dimensional distributions and data using broadcasting.
"""
Distributions.logpdf(dist::BroadcastedDistribution, x) = sum_and_dropdims(logdensityof.(marginals(dist), x), dist.dims)

# Scalar case (required for CUDA)
Distributions.logpdf(dist::ScalarBroadcastedDistribution, x::AbstractArray{<:Real}) = sum_and_dropdims(logdensityof.(materialize(marginals(dist)), x), dist.dims)

# Random Interface

"""
    rand(rng, dist, [dims...])
Sample an array from `dist` of size `(size(marginals)..., dims...)`.
The array type is based on the `rng` and the parameter type of the distribution.
"""
function Base.rand(rng::AbstractRNG, dist::BroadcastedDistribution{T}, dims::Int...) where {T}
    # could probably be generalized by implementing Base.eltype(AbstractVectorizedDistribution)
    A = array_for_rng(rng, T, size(marginals(dist))..., dims...)
    rand!(rng, dist, A)
end

# Scalar case
Base.rand(rng::AbstractRNG, dist::ScalarBroadcastedDistribution, dims::Int...) = rand(rng, materialize(marginals(dist)), dims...)

"""
    rand!(rng, dist, [dims...])
Mutate the array `A` by sampling from `dist`.
"""
Random.rand!(rng::AbstractRNG, dist::BroadcastedDistribution, A::AbstractArray{<:Real}) = rand_kernel!(rng, marginals(dist), A)

# Bijectors

# Each entry might have an individual parameterization of the bijector, also helps with correct device
Bijectors.bijector(dist::BroadcastedDistribution) = BroadcastedBijector(dist.dims, broadcasted(bijector, dist.marginals))

"""
    transformed(dist)
Lazily transforms the distribution type to the unconstrained domain.
"""
Bijectors.transformed(dist::BroadcastedDistribution{T}) where {T} = BroadcastedDistribution(T, dist.dims, broadcasted(transformed, dist.marginals))

Bijectors.link(dist::BroadcastedDistribution, x) = link.(dist.marginals, x)
# Scalar case (required for CUDA)
Bijectors.link(dist::ScalarBroadcastedDistribution, x) = link.(materialize(dist.marginals), x)

Bijectors.invlink(dist::BroadcastedDistribution, y) = invlink.(dist.marginals, y)
# Scalar case (required for CUDA)
Bijectors.invlink(dist::ScalarBroadcastedDistribution, y) = invlink.(materialize(dist.marginals), y)


# Truncation
Distributions.truncated(dist::BroadcastedDistribution{T}, lower, upper) where {T} = BroadcastedDistribution(T, dist.dims, broadcasted(truncated, dist.marginals, lower, upper))
