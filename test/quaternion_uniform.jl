# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

@testset "QuaternionUniform, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred QuaternionUniform(Float64)
    x = @inferred rand(rng, d)
    @test x isa Quaternion{Float64}
    @test abs(x) ≈ 1
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = QuaternionUniform(Float32)
    x = @inferred rand(rng, d)
    @test x isa Quaternion{Float32}
    @test abs(x) ≈ 1
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = QuaternionUniform(Float64)
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Quaternion{Float64}}
    @test reduce(&, abs.(x) .≈ 1)
    # Corner case: all quaternions have been (0,0,0,0) and normalized
    @test reduce(&, x .!= Quaternion(1, 0, 0, 0))
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = QuaternionUniform(Float32)
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Quaternion{Float32}}
    @test reduce(&, abs.(x) .≈ 1)
    # Corner case: all quaternions have been (0,0,0,0) and normalized
    @test reduce(&, x .!= Quaternion(1, 0, 0, 0))
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "QuaternionUniform logdensityof" begin
    kern = QuaternionUniform(Float64)

    @test logdensityof(kern, zero(Quaternion)) == -log(π^2)
    @test logdensityof(kern, fill(zero(Quaternion), 42)) == fill(-log(π^2), 42)
    @test logdensityof(kern, Quaternion(1.0, 2.0, 3.0, 4.0)) == -log(π^2)
end

@testset "QuaternionUniform Bijectors" begin
    @test bijector(QuaternionUniform()) == ZeroIdentity()
end

@testset "QuaternionUniform Transformed, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred transformed(QuaternionUniform(Float64))
    x = @inferred rand(rng, d)
    @test x isa Quaternion{Float64}
    @test abs(x) ≈ 1
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = @inferred transformed(QuaternionUniform(Float32))
    x = @inferred rand(rng, d)
    @test x isa Quaternion{Float32}
    @test abs(x) ≈ 1
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = @inferred transformed(QuaternionUniform(Float64))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Quaternion{Float64}}
    @test reduce(&, abs.(x) .≈ 1)
    # Corner case: all quaternions have been (0,0,0,0) and normalized
    @test reduce(&, x .!= Quaternion(1, 0, 0, 0))
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = @inferred transformed(QuaternionUniform(Float32))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Quaternion{Float32}}
    @test reduce(&, abs.(x) .≈ 1)
    # Corner case: all quaternions have been (0,0,0,0) and normalized
    @test reduce(&, x .!= Quaternion(1, 0, 0, 0))
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "QuaternionUniform Transformed vs. Distributions.jl" begin
    # Compare to Distributions.jl
    kern = transformed(QuaternionUniform(Float64))

    # Identity bijector → no logjac correction
    q = zero(Quaternion)
    @test logdensityof(kern, q) == -log(π^2)
    q = rand(kern)
    @test logdensityof(kern, q) == -log(π^2)
    @test logdensityof(kern, fill(q, 42)) == fill(-log(π^2), 42)

    b = bijector(kern)
    q = rand(kern)
    @test logabsdetjac(b, q) == 0
    q = rand(kern, 42)
    @test logabsdetjac(b, q) == 0
    q = zero(Quaternion)
    @test logabsdetjac(b, q) == 0
    q = fill(q, 42)
    @test logabsdetjac(b, q) == 0
end
