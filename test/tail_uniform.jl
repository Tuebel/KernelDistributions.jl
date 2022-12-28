# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

@testset "TailUniform, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred TailUniform(2.0, 3.0)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = TailUniform(Float32(2), Float32(3))
    x = @inferred rand(rng, d)
    @test x isa Float32
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = TailUniform(2.0, 3.0)
    x = @inferred rand(rng, d, 4_200)
    @test 2 <= minimum(x) < maximum(x) <= 3
    @test x isa AbstractVector{Float64}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = TailUniform(Float32(2), Float32(3))
    x = @inferred rand(rng, d, 4_200)
    @test 2 <= minimum(x) < maximum(x) <= 3
    @test x isa AbstractVector{Float32}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "TailUniform logdensityof" begin
    kern = TailUniform(2.0, 3.0)

    @test logdensityof(kern, -Inf) == 0
    @test logdensityof(kern, 1.9) == 0
    @test logdensityof(kern, 2.0) == 0
    @test logdensityof(kern, 2.1) == 0
    @test logdensityof(kern, 2.9) == 0
    @test logdensityof(kern, 3.0) == 0
    @test logdensityof(kern, 3.1) == 0
    @test logdensityof(kern, Inf) == 0
end

@testset "TailUniform Bijectors" begin
    @test maximum(TailUniform(Float32)) == Inf16
    @test minimum(TailUniform(Float32)) == -Inf16
    @test insupport(TailUniform(Float32), 0)
    @test insupport(TailUniform(Float32), Inf)
    @test insupport(TailUniform(Float32), -Inf)
    @test bijector(TailUniform()) == ZeroIdentity()
end

@testset "TailUniform Transformed, RNG: $rng" for rng in rngs
    # Scalar
    d = transformed(TailUniform(2.0, 3.0))
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = transformed(TailUniform(Float32(2), Float32(3)))
    x = @inferred rand(rng, d)
    @test x isa Float32
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = transformed(TailUniform(2.0, 3.0))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float64}
    @test 2 <= minimum(x) < maximum(x) <= 3
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = transformed(TailUniform(Float32(2), Float32(3)))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float32}
    @test 2 <= minimum(x) < maximum(x) <= 3
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "TailUniform Transformed vs. Distributions.jl" begin
    # Compare to Distributions.jl
    kern = transformed(TailUniform(2.0, 3.0))

    @test logdensityof(kern, -Inf) == 0
    @test logdensityof(kern, 1.9) == 0
    @test logdensityof(kern, 2.0) == 0
    @test logdensityof(kern, 2.1) == 0
    @test logdensityof(kern, 2.9) == 0
    @test logdensityof(kern, 3.0) == 0
    @test logdensityof(kern, 3.1) == 0
    @test logdensityof(kern, Inf) == 0
end
