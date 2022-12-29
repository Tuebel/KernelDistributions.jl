# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

@testset "SmoothExponential, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred SmoothExponential(3.0, 7.0, 2.0, 0.1)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = SmoothExponential(Float32(2))
    x = @inferred rand(rng, d)
    @test x isa Float32
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = SmoothExponential(2.0)
    x = @inferred rand(rng, d, 4_200)
    @test 0 <= minimum(x) < maximum(x)
    @test x isa AbstractVector{Float64}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = SmoothExponential(Float32(2))
    x = @inferred rand(rng, d, 4_200)
    @test 0 <= minimum(x) < maximum(x)
    @test x isa AbstractVector{Float32}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "SmoothExponential vs. Distributions.jl" begin
    dist = Exponential(3.0)
    kern = SmoothExponential(3.0)

    @test logdensityof(dist, 1.0) == logdensityof(kern, 1.0)
    @test logdensityof(dist, 0.0) == logdensityof(kern, 0.0)
    @test logdensityof(dist, -1.0) == logdensityof(kern, -1.0)
    @test logcdf(dist, 1.0) == logcdf(kern, 1.0)
    @test logcdf(dist, 0.0) == logcdf(kern, 0.0)
    @test logcdf(dist, -1.0) == logcdf(kern, -1.0)
    @test_throws DomainError invlogcdf(dist, 0.1)
    @test invlogcdf(dist, 0.0) == invlogcdf(kern, 0.0)
    @test invlogcdf(dist, -1.0) == invlogcdf(kern, -1.0)
end

@testset "SmoothExponential Bijectors" begin
    @test maximum(SmoothExponential(Float32)) == Inf16
    @test minimum(SmoothExponential(Float32)) == 0
    @test insupport(SmoothExponential(Float32), 0)
    @test insupport(SmoothExponential(Float32), Inf)
    @test !insupport(SmoothExponential(Float32), -eps(Float32))
    @test bijector(SmoothExponential()) == bijector(Exponential())
end

@testset "SmoothExponential Truncated, RNG: $rng" for rng in rngs
    # Scalar
    d = truncated(SmoothExponential(2.0), 1.0, 2.0)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = truncated(SmoothExponential(Float32(2)), Float32(1), Float32(2))
    x = @inferred rand(rng, d)
    @test x isa Float32
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = truncated(SmoothExponential(2.0), 1.0, 2.0)
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float64}
    @test 1 <= minimum(x) < maximum(x) <= 2
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    # Fails for Float32
    d = truncated(SmoothExponential(Float32(2)), Float32(1), Float32(2))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float32}
    @test 1 <= minimum(x) < maximum(x) <= 2
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "SmoothExponential Truncated vs. Distributions.jl" begin
    # Compare to Distributions.jl
    dist = truncated(Exponential(3.0), 2.0, 2.0)
    kern = truncated(SmoothExponential(3.0), 2.0, 2.0)

    @test logdensityof(kern, 0.9) == logdensityof(dist, 0.9)
    @test logdensityof(kern, 1.0) == logdensityof(dist, 1.0)
    @test logdensityof(kern, 1.5) == logdensityof(dist, 1.5)
    @test logdensityof(kern, 2.0) == logdensityof(dist, 2.0)
    @test logdensityof(kern, 2.1) == logdensityof(dist, 2.1)
end

@testset "SmoothExponential Transformed, RNG: $rng" for rng in rngs
    # Scalar
    d = transformed(SmoothExponential(2.0))
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = transformed(SmoothExponential(Float32(2)))
    x = @inferred rand(rng, d)
    @test x isa Float32
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = transformed(SmoothExponential(2.0))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float64}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = transformed(SmoothExponential(Float32(2)))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float32}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "SmoothExponential Transformed vs. Distributions.jl" begin
    # Compare to Distributions.jl
    dist = transformed(Exponential(3.0))
    kern = transformed(SmoothExponential(3.0))

    @test logdensityof(kern, 0.9) == logdensityof(dist, 0.9)
    @test logdensityof(kern, 1.0) == logdensityof(dist, 1.0)
    @test logdensityof(kern, 1.5) == logdensityof(dist, 1.5)
    @test logdensityof(kern, 2.0) == logdensityof(dist, 2.0)
    @test logdensityof(kern, 2.1) == logdensityof(dist, 2.1)
end
