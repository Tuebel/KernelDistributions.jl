# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

@test KernelNormal(1, 2.0) |> show |> isnothing

@testset "KernelNormal, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred KernelNormal(2, 1.0)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = KernelNormal(2, Float32(1))
    x = @inferred rand(rng, d)
    @test x isa Float32
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = KernelNormal(2, 1.1)
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float64}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = KernelNormal(2, Float32(1))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float32}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "KernelNormal vs. Distributions.jl" begin
    dist = Normal(1.0, 3.0)
    kern = KernelNormal(1.0, 3.0)

    @test logdensityof(dist, 1.0) == logdensityof(kern, 1.0)
    @test logdensityof(dist, 0.0) == logdensityof(kern, 0.0)
    @test logdensityof(dist, -1.0) == logdensityof(kern, -1.0)
    @test logcdf(dist, 1.0) == logcdf(kern, 1.0)
    @test logcdf(dist, 0.0) == logcdf(kern, 0.0)
    @test logcdf(dist, -1.0) == logcdf(kern, -1.0)
    @test isnan(invlogcdf(kern, 0.1))
    @test invlogcdf(dist, 0.0) == invlogcdf(kern, 0.0)
    @test invlogcdf(dist, -1.0) == invlogcdf(kern, -1.0)
end

@testset "KernelNormal Bijectors" begin
    @test maximum(KernelNormal(Float32)) == Inf16
    @test minimum(KernelNormal(Float32)) == -Inf16
    @test insupport(KernelNormal(Float32), 0)
    @test insupport(KernelNormal(Float32), Inf)
    @test insupport(KernelNormal(Float32), -Inf)
    @test bijector(KernelNormal()) == ZeroIdentity()
end

@testset "KernelNormal Truncated, RNG: $rng" for rng in rngs
    # Scalar
    d = truncated(KernelNormal(2, 1.1), 1.0, 2.0)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = truncated(KernelNormal(2, Float32(1)), Float32(1), Float32(2))
    x = @inferred rand(rng, d)
    @test x isa Float32
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = truncated(KernelNormal(2, 1.1), 1.0, 2.0)
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float64}
    @test 1 <= minimum(x) < maximum(x) <= 2
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = truncated(KernelNormal(2, Float32(1)), Float32(1), Float32(2))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float32}
    @test 1 <= minimum(x) < maximum(x) <= 2
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "KernelNormal Truncated vs. Distributions.jl" begin
    # Compare to Distributions.jl
    dist = truncated(Normal(2, 1.1), 1.0, 2.0)
    kern = truncated(KernelNormal(2, 1.1), 1.0, 2.0)

    @test logdensityof(kern, 0.9) == logdensityof(dist, 0.9)
    @test logdensityof(kern, 1.0) == logdensityof(dist, 1.0)
    @test logdensityof(kern, 1.5) == logdensityof(dist, 1.5)
    @test logdensityof(kern, 2.0) == logdensityof(dist, 2.0)
    @test logdensityof(kern, 2.1) == logdensityof(dist, 2.1)
end

@testset "KernelNormal Transformed, RNG: $rng" for rng in rngs
    # Scalar
    d = transformed(KernelNormal(2, 1.1))
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = transformed(KernelNormal(2, Float32(1)))
    x = @inferred rand(rng, d)
    @test x isa Float32
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = transformed(KernelNormal(2, 1.1))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float64}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = transformed(KernelNormal(2, Float32(1)))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float32}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "KernelNormal Transformed vs. Distributions.jl" begin
    # Compare to Distributions.jl
    dist = transformed(Normal(2, 1.1))
    kern = transformed(KernelNormal(2, 1.1))

    @test logdensityof(kern, 0.9) == logdensityof(dist, 0.9)
    @test logdensityof(kern, 1.0) == logdensityof(dist, 1.0)
    @test logdensityof(kern, 1.5) == logdensityof(dist, 1.5)
    @test logdensityof(kern, 2.0) == logdensityof(dist, 2.0)
    @test logdensityof(kern, 2.1) == logdensityof(dist, 2.1)
end
