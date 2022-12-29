# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

accurate_exp(min, max, θ, σ, z) = exp(-z / θ + (σ / θ)^2 / 2) / (θ * (exp(-min / θ) - exp(-max / θ)))
function accurate_erf(min, max, θ, σ, z)
    sqrt2σ = sqrt2 * σ
    common = σ / (sqrt2 * θ) - z / sqrt2σ
    lower = min / sqrt2σ
    upper = max / sqrt2σ
    erf(common + lower, common + upper) / 2
end
smooth_pdf(min, max, θ, σ, z) = accurate_exp(min, max, θ, σ, z) * accurate_erf(min, max, θ, σ, z)
# WARN numerically not stable for values far outside [min,max]
smooth_logpdf = log ∘ smooth_pdf

@testset "SmoothExponential, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred SmoothExponential(3.0, 7.0, 2.0, 0.1)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = SmoothExponential(3.0f0, 7.0f0, 2.0f0, 0.1f0)
    x = @inferred rand(rng, d)
    @test x isa Float32
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = @inferred SmoothExponential(3.0, 7.0, 2.0, 0.1)
    x = @inferred rand(rng, d, 4_200)
    @test 0 <= minimum(x) < maximum(x) <= 10
    @test x isa AbstractVector{Float64}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = SmoothExponential(3.0f0, 7.0f0, 2.0f0, 0.1f0)
    x = @inferred rand(rng, d, 4_200)
    @test 0 <= minimum(x) < maximum(x) <= 10
    @test x isa AbstractVector{Float32}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "SmoothExponential logdensityof" begin
    kern = @inferred SmoothExponential(3.0, 7.0, 2.0, 0.1)

    @test logdensityof(kern, 1.0) ≈ smooth_logpdf(3.0, 7.0, 2.0, 0.1, 1.0)
    @test logdensityof(kern, 3.0) ≈ smooth_logpdf(3.0, 7.0, 2.0, 0.1, 3.0)
    @test logdensityof(kern, 7.0) ≈ smooth_logpdf(3.0, 7.0, 2.0, 0.1, 7.0)
    @test logdensityof(kern, 8.0) ≈ smooth_logpdf(3.0, 7.0, 2.0, 0.1, 8.0)
end

@testset "SmoothExponential Bijectors" begin
    d = SmoothExponential(3.0f0, 7.0f0, 2.0f0, 0.1f0)

    @test maximum(d) == typemax(Float32)
    @test minimum(d) == typemin(Float32)
    @test insupport(d, -Inf32)
    @test insupport(d, 0)
    @test insupport(d, Inf32)
    @test bijector(d) == ZeroIdentity()
end

@testset "SmoothExponential Transformed, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred SmoothExponential(3.0, 7.0, 2.0, 0.1)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = SmoothExponential(3.0f0, 7.0f0, 2.0f0, 0.1f0)
    x = @inferred rand(rng, d)
    @test x isa Float32
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = @inferred SmoothExponential(3.0, 7.0, 2.0, 0.1)
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float64}
    @test 0 <= minimum(x) < maximum(x) <= 10
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = SmoothExponential(3.0f0, 7.0f0, 2.0f0, 0.1f0)
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float32}
    @test 0 <= minimum(x) < maximum(x) <= 10
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "SmoothExponential Transformed logdensityof" begin
    kern = @inferred transformed(SmoothExponential(3.0, 7.0, 2.0, 0.1))

    @test logdensityof(kern, 1.0) ≈ smooth_logpdf(3.0, 7.0, 2.0, 0.1, 1.0)
    @test logdensityof(kern, 3.0) ≈ smooth_logpdf(3.0, 7.0, 2.0, 0.1, 3.0)
    @test logdensityof(kern, 7.0) ≈ smooth_logpdf(3.0, 7.0, 2.0, 0.1, 7.0)
    @test logdensityof(kern, 8.0) ≈ smooth_logpdf(3.0, 7.0, 2.0, 0.1, 8.0)
end
