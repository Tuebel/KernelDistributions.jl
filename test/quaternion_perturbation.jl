# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

σ = 0.01
quatpert_logpdf(q::AdditiveQuaternion) = sum(logdensityof.(KernelNormal(0, σ), imag_part(q.q) .* 2))
not_identity(q::AdditiveQuaternion) = q.q != Quaternion(1, 0, 0, 0)

@testset "QuaternionPerturbation, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred QuaternionPerturbation(σ)
    x = @inferred rand(rng, d)
    @test x isa AdditiveQuaternion{Float64}
    @test abs(x.q) ≈ 1
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = QuaternionPerturbation(Float32(σ))
    x = @inferred rand(rng, d)
    @test x isa AdditiveQuaternion{Float32}
    @test abs(x.q) ≈ 1
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = QuaternionPerturbation(σ)
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{AdditiveQuaternion{Float64}}
    @test reduce(&, abs.(x) .≈ 1)
    # Corner case: all quaternions have been (0,0,0,0) and normalized
    @test reduce(&, not_identity.(x))
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = QuaternionPerturbation(Float32(σ))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{AdditiveQuaternion{Float32}}
    @test reduce(&, abs.(x) .≈ 1)
    # Corner case: all quaternions have been (0,0,0,0) and normalized
    @test reduce(&, not_identity.(x))
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "QuaternionPerturbation logdensityof" begin
    kern = QuaternionPerturbation(σ)

    q = AdditiveQuaternion(zero(Quaternion))
    @test logdensityof(kern, q) == quatpert_logpdf(q)
    q = rand(kern)
    @test logdensityof(kern, q) == quatpert_logpdf(q)
end

@testset "QuaternionPerturbation Bijectors" begin
    @test bijector(QuaternionPerturbation()) == ZeroIdentity()
end

@testset "QuaternionPerturbation Transformed, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred transformed(QuaternionPerturbation(σ))
    x = @inferred rand(rng, d)
    @test x isa AdditiveQuaternion{Float64}
    @test abs(x.q) ≈ 1
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = @inferred transformed(QuaternionPerturbation(Float32(σ)))
    x = @inferred rand(rng, d)
    @test x isa AdditiveQuaternion{Float32}
    @test abs(x.q) ≈ 1
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = @inferred transformed(QuaternionPerturbation(σ))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{AdditiveQuaternion{Float64}}
    @test reduce(&, abs.(x) .≈ 1)
    # Corner case: all quaternions have been (0,0,0,0) and normalized
    @test reduce(&, not_identity.(x))
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = @inferred transformed(QuaternionPerturbation(Float32(σ)))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{AdditiveQuaternion{Float32}}
    @test reduce(&, abs.(x) .≈ 1)
    # Corner case: all quaternions have been (0,0,0,0) and normalized
    @test reduce(&, not_identity.(x))
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "QuaternionPerturbation Transformed vs. Distributions.jl" begin
    # Compare to Distributions.jl
    kern = transformed(QuaternionPerturbation(σ))

    # Identity bijector → no logjac correction
    q = AdditiveQuaternion(zero(Quaternion))
    @test logdensityof(kern, q) == quatpert_logpdf(q)
    q = rand(kern)
    @test logdensityof(kern, q) == quatpert_logpdf(q)
    @test logdensityof(kern, fill(q, 42)) == quatpert_logpdf.(fill(q, 42))

    b = bijector(kern)
    q = rand(kern)
    @test logabsdetjac(b, q) == 0
    q = rand(kern, 42)
    @test logabsdetjac(b, q) == 0
    q = AdditiveQuaternion(zero(Quaternion))
    @test logabsdetjac(b, q) == 0
    q = fill(q, 42)
    @test logabsdetjac(b, q) == 0
end
