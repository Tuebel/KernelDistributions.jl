# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    CircularUniform
Generates uniform random numbers ∈ [0,2π].
If the lower/upper bound is exceeded the value continues at the other bound (2π+0.1=0.1).
"""
struct CircularUniform{T<:Real} <: AbstractKernelDistribution{T,Continuous} end

CircularUniform(::Type{T}=Float32) where {T} = CircularUniform{T}()

Base.show(io::IO, ::CircularUniform{T}) where {T} = print(io, "CircularUniform{$(T)}")

Distributions.logpdf(dist::CircularUniform{T}, x) where {T} = insupport(dist, x) ? -log(T(2π)) : -typemax(T)

rand_kernel(rng::AbstractRNG, ::CircularUniform{T}) where {T} = T(2π) * rand(rng, T)

Base.maximum(::CircularUniform{T}) where {T} = T(2π)
Base.minimum(::CircularUniform{T}) where {T} = zero(T)
Bijectors.bijector(::CircularUniform) = Circular()
Distributions.insupport(dist::CircularUniform, x::Real) = minimum(dist) <= x <= maximum(dist)


"""
    Circular
Transform ℝ → [0,2π)
"""
struct Circular <: Bijector end

"""
    transform(Circular, x)
Transform from [0,2π] to ℝ.
In theory inverse of mod does not exist, in practice the same value is returned, since `[0,2π] ∈ ℝ`
"""
Bijectors.transform(::Circular, x) = x

"""
    transform(Circular, y)
Uses `mod2pi` to transform ℝ to [0,2π].
"""
Bijectors.transform(::Inverse{Circular}, y) = mod2pi.(y)

"""
    logabsdetjac(Circular, x)
mod2pi will not be zero for n*2*π, thus the discontinuity will not be reached.
Thus, the log Jacobian is always 0. 
"""
Bijectors.logabsdetjac(::Circular, x) = zero(x)
ChangesOfVariables.with_logabsdet_jacobian(b::Circular, x) = b(x), logabsdetjac(b, x)

Bijectors.logabsdetjac(::Inverse{<:Circular}, y) = zero(y)
ChangesOfVariables.with_logabsdet_jacobian(b::Inverse{Circular}, y) = b(y), logabsdetjac(b, y)
