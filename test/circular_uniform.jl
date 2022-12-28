# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

@testset "CircularUniform, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred CircularUniform(Float64)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = CircularUniform(Float16)
    x = @inferred rand(rng, d)
    @test x isa Float16
    l = @inferred logdensityof(d, x)
    @test l isa Float16

    # Array
    d = CircularUniform(Float64)
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float64}
    @test 0 < minimum(x) < maximum(x) < 2π
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = CircularUniform(Float16)
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float16}
    @test 0 < minimum(x) < maximum(x) < 2π
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float16}
end

@testset "CircularUniform logdensityof" begin
    kern = CircularUniform(Float64)

    @test logdensityof(kern, -0.01) == -Inf
    @test logdensityof(kern, 0.0) == log(1 / 2π)
    @test logdensityof(kern, 1.0) == log(1 / 2π)
    @test logdensityof(kern, 2π) == log(1 / 2π)
    @test logdensityof(kern, 2π + 0.01) == -Inf
end

@testset "CircularUniform Bijectors" begin
    @test maximum(CircularUniform(Float16)) == Float16(2π)
    @test minimum(CircularUniform(Float16)) == 0
    @test insupport(CircularUniform(Float16), 0)
    @test insupport(CircularUniform(Float16), Float16(2π))
    @test !insupport(CircularUniform(Float16), -0.01)
    @test !insupport(CircularUniform(Float16), Float16(2π + 0.01))
    @test bijector(CircularUniform()) == Circular()
end

@testset "CircularUniform Transformed, RNG: $rng" for rng in rngs
    # Scalar
    d = transformed(CircularUniform(Float64))
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = transformed(CircularUniform(Float16))
    x = @inferred rand(rng, d)
    @test x isa Float16
    l = @inferred logdensityof(d, x)
    @test l isa Float16

    # Array
    d = transformed(CircularUniform(Float64))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float64}
    @test 0 < minimum(x) < maximum(x) < 2π
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = transformed(CircularUniform(Float16))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float16}
    @test 0 < minimum(x) < maximum(x) < 2π
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float16}
end

@testset "CircularUniform Transformed vs. Distributions.jl" begin
    # Compare to Distributions.jl
    dist = transformed(Uniform(2.0, 3.0))
    kern = transformed(CircularUniform(Float64))

    @test logdensityof(kern, -Inf) == -Inf
    @test logdensityof(kern, -0.01) == log(1 / 2π)
    @test logdensityof(kern, 0.0) == log(1 / 2π)
    @test logdensityof(kern, 1.0) == log(1 / 2π)
    @test logdensityof(kern, 2π) == log(1 / 2π)
    @test logdensityof(kern, 2π + 0.01) == log(1 / 2π)
    @test logdensityof(kern, Inf) == -Inf
end
