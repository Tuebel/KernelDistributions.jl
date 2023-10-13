# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Bijectors
using KernelDistributions
using Random
using Test

@test QuaternionNormal(Float64) |> show |> isnothing

# For now only CPU support
rng = Random.default_rng()
# Likelihood for μ=(1,0,0,0), σ=0, and x=(1,0,0,0)
ℓ_unit = logdensityof(KernelNormal(), 0) * 3

@testset "QuaternionUniform rand" begin
    # Scalar
    d = @inferred QuaternionNormal(Float64)
    x = @inferred rand(rng, d)
    @test x isa Quaternion{Float64}
    @test abs(x) ≈ 1
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = QuaternionNormal(sign(Quaternion(1.0f0, 1.0f0, 0, 0)), 0.1f0)
    x = @inferred rand(rng, d)
    @test x isa Quaternion{Float32}
    @test abs(x) ≈ 1
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = QuaternionNormal(Float64)
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Quaternion{Float64}}
    @test reduce(&, abs.(x) .≈ 1)
    # Corner case: all quaternions have been (0,0,0,0) and normalized
    @test reduce(&, x .!= Quaternion(1, 0, 0, 0))
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = QuaternionNormal(Float32)
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Quaternion{Float32}}
    @test reduce(&, abs.(x) .≈ 1)
    # Corner case: all quaternions have been (0,0,0,0) and normalized
    @test reduce(&, x .!= Quaternion(1, 0, 0, 0))
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "QuaternionNormal logdensityof" begin
    d = QuaternionNormal(Float64)
    # robust implementation
    ℓ = @inferred logdensityof(d, zero(Quaternion))
    @test logdensityof(d, zero(Quaternion)) ≈ ℓ_unit
    ℓ = @inferred logdensityof(d, one(Quaternion))
    @test logdensityof(d, zero(Quaternion)) ≈ ℓ_unit
    @test logdensityof(d, fill(zero(Quaternion), 42)) ≈ fill(ℓ_unit, 42)
    @test !(logdensityof(d, Quaternion(1.0, 2.0, 3.0, 4.0)) ≈ ℓ_unit)
end

@testset "QuaternionNormal Bijectors" begin
    @test bijector(QuaternionNormal()) == ZeroIdentity()
end

@testset "QuaternionNormal Transformed" begin
    # Scalar
    d = @inferred transformed(QuaternionNormal(Float64))
    x = @inferred rand(rng, d)
    @test x isa Quaternion{Float64}
    @test abs(x) ≈ 1
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = @inferred transformed(QuaternionNormal(Float32))
    x = @inferred rand(rng, d)
    @test x isa Quaternion{Float32}
    @test abs(x) ≈ 1
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = @inferred transformed(QuaternionNormal(Float64))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Quaternion{Float64}}
    @test reduce(&, abs.(x) .≈ 1)
    # Corner case: all quaternions have been (0,0,0,0) and normalized
    @test reduce(&, x .!= Quaternion(1, 0, 0, 0))
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = @inferred transformed(QuaternionNormal(Float32))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Quaternion{Float32}}
    @test reduce(&, abs.(x) .≈ 1)
    # Corner case: all quaternions have been (0,0,0,0) and normalized
    @test reduce(&, x .!= Quaternion(1, 0, 0, 0))
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "QuaternionNormal Transformed vs. Distributions.jl" begin
    # Compare to Distributions.jl
    d = transformed(QuaternionNormal(Float64))

    # Identity bijector → no logjac correction
    q = zero(Quaternion)
    @test logdensityof(d, q) ≈ ℓ_unit
    q = rand(d)
    @test !(logdensityof(d, q) ≈ ℓ_unit)
    @test logdensityof(d, fill(one(Quaternion), 42)) ≈ fill(ℓ_unit, 42)

    b = bijector(d)
    q = rand(d)
    @test logabsdetjac(b, q) == 0
    q = rand(d, 42)
    @test logabsdetjac(b, q) == 0
    q = zero(Quaternion)
    @test logabsdetjac(b, q) == 0
    @test logabsdetjac(inverse(b), q) == 0
    q = fill(q, 42)
    @test logabsdetjac(b, q) == 0
    @test logabsdetjac(inverse(b), q) == 0
end
