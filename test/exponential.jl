# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

@testset "KernelExponential, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred KernelExponential(2.0)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = KernelExponential(Float16(2))
    x = @inferred rand(rng, d)
    @test x isa Float16
    l = @inferred logdensityof(d, x)
    @test l isa Float16

    # Array
    d = KernelExponential(2.0)
    x = @inferred rand(rng, d, 4_200)
    @test 0 <= minimum(x) < maximum(x)
    @test x isa AbstractVector{Float64}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = KernelExponential(Float16(2))
    x = @inferred rand(rng, d, 4_200)
    @test 0 <= minimum(x) < maximum(x)
    @test x isa AbstractVector{Float16}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float16}
end

@testset "KernelExponential vs. Distributions.jl" begin
    dist = Exponential(3.0)
    kern = KernelExponential(3.0)

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

@testset "KernelExponential Bijectors" begin
    @test maximum(KernelExponential(Float16)) == Inf16
    @test minimum(KernelExponential(Float16)) == 0
    @test insupport(KernelExponential(Float16), 0)
    @test insupport(KernelExponential(Float16), Inf)
    @test !insupport(KernelExponential(Float16), -eps(Float16))
    @test bijector(KernelExponential()) == bijector(Exponential())
end

@testset "KernelExponential Truncated, RNG: $rng" for rng in rngs
    # Scalar
    d = truncated(KernelExponential(2.0), 1.0, 2.0)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = truncated(KernelExponential(Float16(2)), Float16(1), Float16(2))
    x = @inferred rand(rng, d)
    @test x isa Float16
    l = @inferred logdensityof(d, x)
    @test l isa Float16

    # Array
    d = truncated(KernelExponential(2.0), 1.0, 2.0)
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float64}
    @test 1 <= minimum(x) < maximum(x) <= 2
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    # Fails for Float32
    d = truncated(KernelExponential(Float32(2)), Float32(1), Float32(2))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float32}
    @test 1 <= minimum(x) < maximum(x) <= 2
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "KernelExponential Truncated vs. Distributions.jl" begin
    # Compare to Distributions.jl
    dist = truncated(Exponential(3.0), 2.0, 2.0)
    kern = truncated(KernelExponential(3.0), 2.0, 2.0)

    @test logdensityof(kern, 0.9) == logdensityof(dist, 0.9)
    @test logdensityof(kern, 1.0) == logdensityof(dist, 1.0)
    @test logdensityof(kern, 1.5) == logdensityof(dist, 1.5)
    @test logdensityof(kern, 2.0) == logdensityof(dist, 2.0)
    @test logdensityof(kern, 2.1) == logdensityof(dist, 2.1)
end

@testset "KernelExponential Transformed, RNG: $rng" for rng in rngs
    # Scalar
    d = transformed(KernelExponential(2.0))
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = transformed(KernelExponential(Float16(2)))
    x = @inferred rand(rng, d)
    @test x isa Float16
    l = @inferred logdensityof(d, x)
    @test l isa Float16

    # Array
    d = transformed(KernelExponential(2.0))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float64}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = transformed(KernelExponential(Float16(2)))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float16}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float16}
end

@testset "KernelExponential Transformed vs. Distributions.jl" begin
    # Compare to Distributions.jl
    dist = transformed(Exponential(3.0))
    kern = transformed(KernelExponential(3.0))

    @test logdensityof(kern, 0.9) == logdensityof(dist, 0.9)
    @test logdensityof(kern, 1.0) == logdensityof(dist, 1.0)
    @test logdensityof(kern, 1.5) == logdensityof(dist, 1.5)
    @test logdensityof(kern, 2.0) == logdensityof(dist, 2.0)
    @test logdensityof(kern, 2.1) == logdensityof(dist, 2.1)
end
