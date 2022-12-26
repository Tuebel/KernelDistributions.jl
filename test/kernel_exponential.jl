# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

@testset "KernelExponential, RNG: $rng" for rng in rngs
    # Scalar
    d = KernelExponential(1.0)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = KernelExponential(Float16(1))
    x = @inferred rand(rng, d)
    @test x isa Float16
    l = @inferred logdensityof(d, x)
    @test l isa Float16

    # Array
    d = KernelExponential(1.0)
    x = @inferred rand(rng, d, 42)
    @test x isa AbstractVector{Float64}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = KernelExponential(Float16(1))
    x = @inferred rand(rng, d, 42)
    @test x isa AbstractVector{Float16}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float16}
end

@testset "KernelExponential vs. Distributions.jl" begin
    dist_exp = Exponential(3.0)
    kern_exp = Exponential(3.0)

    @test logdensityof(dist_exp, 1.0) == logdensityof(kern_exp, 1.0)
    @test logdensityof(dist_exp, 0.0) == logdensityof(kern_exp, 0.0)
    @test logdensityof(dist_exp, -1.0) == logdensityof(kern_exp, -1.0)
    @test logcdf(dist_exp, 1.0) == logcdf(kern_exp, 1.0)
    @test logcdf(dist_exp, 0.0) == logcdf(kern_exp, 0.0)
    @test logcdf(dist_exp, -1.0) == logcdf(kern_exp, -1.0)
    @test_throws DomainError invlogcdf(dist_exp, 0.1)
    @test invlogcdf(dist_exp, 0.0) == invlogcdf(kern_exp, 0.0)
    @test invlogcdf(dist_exp, -1.0) == invlogcdf(kern_exp, -1.0)
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
    d = truncated(KernelExponential(1.0), 1.0, 2.0)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = truncated(KernelExponential(Float16(1)), Float16(1), Float16(2))
    x = @inferred rand(rng, d)
    @test x isa Float16
    l = @inferred logdensityof(d, x)
    @test l isa Float16

    # Array
    d = truncated(KernelExponential(1.0), 1.0, 2.0)
    x = @inferred rand(rng, d, 42)
    @test x isa AbstractVector{Float64}
    # BUG minimum of CuArray is 0.0 even though no 0.0 is in the CuArray?
    @test 1 < minimum(Array(x)) < maximum(x) < 2
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = truncated(KernelExponential(Float16(1)), Float16(1), Float16(2))
    x = @inferred rand(rng, d, 42)
    @test x isa AbstractVector{Float16}
    # BUG minimum of CuArray is 0.0 even though no 0.0 is in the CuArray?
    @test 1 < minimum(Array(x)) < maximum(x) < 2
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float16}
end

@testset "KernelExponential Truncated vs. Distributions.jl" begin
    # Compare to Distributions.jl
    dist_exp = truncated(Exponential(3.0), 1.0, 2.0)
    kern_exp = truncated(KernelExponential(3.0), 1.0, 2.0)

    @test logdensityof(kern_exp, 0.9) == logdensityof(dist_exp, 0.9)
    @test logdensityof(kern_exp, 1.0) == logdensityof(dist_exp, 1.0)
    @test logdensityof(kern_exp, 1.5) == logdensityof(dist_exp, 1.5)
    @test logdensityof(kern_exp, 2.0) == logdensityof(dist_exp, 2.0)
    @test logdensityof(kern_exp, 2.1) == logdensityof(dist_exp, 2.1)
end

@testset "KernelExponential Transformed, RNG: $rng" for rng in rngs
    # Scalar
    d = transformed(KernelExponential(1.0))
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = transformed(KernelExponential(Float16(1)))
    x = @inferred rand(rng, d)
    @test x isa Float16
    l = @inferred logdensityof(d, x)
    @test l isa Float16

    # Array
    d = transformed(KernelExponential(1.0))
    x = @inferred rand(rng, d, 3)
    @test x isa AbstractVector{Float64}
    # TODO CuArray minimum = 0.0 bug not here?
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = transformed(KernelExponential(Float16(1)))
    x = @inferred rand(rng, d, 3)
    @test x isa AbstractVector{Float16}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float16}
end

@testset "KernelExponential Transformed vs. Distributions.jl" begin
    # Compare to Distributions.jl
    dist_exp = transformed(Exponential(3.0))
    kern_exp = transformed(KernelExponential(3.0))

    @test logdensityof(kern_exp, 0.9) == logdensityof(dist_exp, 0.9)
    @test logdensityof(kern_exp, 1.0) == logdensityof(dist_exp, 1.0)
    @test logdensityof(kern_exp, 1.5) == logdensityof(dist_exp, 1.5)
    @test logdensityof(kern_exp, 2.0) == logdensityof(dist_exp, 2.0)
    @test logdensityof(kern_exp, 2.1) == logdensityof(dist_exp, 2.1)
end
