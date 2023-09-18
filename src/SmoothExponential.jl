# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    SmoothExponential(min, max, β, σ)
Smooth truncated exponential distribution by convolving the exponential with a normal distribution: Smooth = Exp ⋆ Normal
This results in smooth min and max limits and a definition on ℝ instead of ℝ⁺

Does not support `truncated` of Distributions.jl since it is a smooth truncation of the exponential distribution.
"""
struct SmoothExponential{T<:Real} <: AbstractKernelDistribution{T,Continuous}
    min::T
    max::T
    β::T
    σ::T
end

Base.show(io::IO, dist::SmoothExponential{T}) where {T} = print(io, "SmoothExponential{$(T)}, min: $(dist.min), max: $(dist.max), β: $(dist.β), σ: $(dist.σ)")

# Accurate version uses lower and upper bound
accurate_normalization(d::SmoothExponential) = -logsubexp(-d.min / d.β, -d.max / d.β)
accurate_factor(d::SmoothExponential, x) = (-x / d.β + (d.σ / d.β)^2 / 2) - log(d.β) + accurate_normalization(d) + loghalf
function accurate_logerf(d::SmoothExponential{T}, x) where {T}
    invsqrt2σ = inv(my_sqrt2(T) * d.σ)
    lower = (d.min + d.σ^2 / d.β - x) * invsqrt2σ
    upper = (d.max + d.σ^2 / d.β - x) * invsqrt2σ
    my_logerf(lower, upper)
end

# Re-implementation of LogExpFunctions logerf which does not work with CUDA & Julia > 1.9 https://github.com/JuliaGPU/GPUCompiler.jl/issues/384 
function my_logerf(a::T, b::T) where {T<:Real}
    if abs(a) ≤ my_invsqrt2(T) && abs(b) ≤ my_invsqrt2(T)
        return log(erf(a, b))
    elseif b > a > 0
        return logerfc(a) + log1mexp(logerfc(b) - logerfc(a))
    elseif a < b < 0
        return logerfc(-b) + LogExpFunctions.log1mexp(logerfc(-a) - logerfc(-b))
    else
        return log(erf(a, b))
    end
end
my_sqrt2(::Type{T}) where {T<:Real} = T(sqrt2)
my_invsqrt2(::Type{T}) where {T<:Real} = T(invsqrt2)

# See my (Tim Redick) dissertation for the derivation.
Distributions.logpdf(dist::SmoothExponential{T}, x) where {T} = insupport(dist, x) ? accurate_factor(dist, x) + accurate_logerf(dist, x) : typemin(T)

# Exponential convoluted with normal: Sample from exponential and then add noise of normal
function rand_kernel(rng::AbstractRNG, dist::SmoothExponential{T}) where {T}
    # Distributions.jl truncated this fails to compile on RTX3080 so use naive implementation
    μ = rand(rng, KernelExponential(dist.β)) + dist.min
    # BUG on RTX 3080 sometimes samples forever
    # while μ > dist.max
    #     μ = rand(rng, KernelExponential(dist.β)) + dist.min
    # end
    rand(rng, KernelNormal(μ, dist.σ))
end

# Compared to a regular exponential distribution, this one is defined on ℝ 😃
Base.maximum(::SmoothExponential{T}) where {T} = typemax(T)
Base.minimum(::SmoothExponential{T}) where {T} = typemin(T)
Bijectors.bijector(::SmoothExponential) = ZeroIdentity()
# Numerical issues if min≈max. Return limit
Distributions.insupport(dist::SmoothExponential{T}, x::Real) where {T} = abs(dist.max - dist.min) < eps(T) ? false : true
