# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

@test KernelDirac(2.0) |> show |> isnothing

@testset "KernelDirac, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred KernelDirac(2.0)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = KernelDirac(Float32(2))
    x = @inferred rand(rng, d)
    @test x isa Float32
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = KernelDirac(2.0)
    x = @inferred rand(rng, d, 4_200)
    @test minimum(x) == 2 == maximum(x)
    @test x isa AbstractVector{Float64}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = KernelDirac(Float32(2))
    x = @inferred rand(rng, d, 4_200)
    @test minimum(x) == 2 == maximum(x)
    @test x isa AbstractVector{Float32}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "KernelDirac vs. Distributions" begin
    dist = Dirac(3.0)
    kern = KernelDirac(3.0)

    @test logdensityof(dist, 3.0) == logdensityof(kern, 3.0)
    @test logdensityof(dist, 0.0) == logdensityof(kern, 0.0)
end

@testset "KernelDirac Bijectors" begin
    @test maximum(KernelDirac(Float32(3))) == 3
    @test minimum(KernelDirac(Float32(3))) == 3
    @test insupport(KernelDirac(Float32(3)), 3)
    @test !insupport(KernelDirac(Float32(3)), Inf)
    @test !insupport(KernelDirac(Float32(3)), 2.9)
    @test bijector(KernelDirac(2)) == ZeroIdentity()
end
