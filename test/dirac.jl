# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

@testset "KernelDirac, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred KernelDirac(2.0)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = KernelDirac(Float16(2))
    x = @inferred rand(rng, d)
    @test x isa Float16
    l = @inferred logdensityof(d, x)
    @test l isa Float16

    # Array
    d = KernelDirac(2.0)
    x = @inferred rand(rng, d, 3)
    @test x isa AbstractVector{Float64}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = KernelDirac(Float16(2))
    x = @inferred rand(rng, d, 3)
    @test x isa AbstractVector{Float16}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float16}
end

@testset "KernelDirac vs. Distributions" begin
    dist = Dirac(3.0)
    kern = KernelDirac(3.0)

    @test logdensityof(dist, 3.0) == logdensityof(kern, 3.0)
    @test logdensityof(dist, 0.0) == logdensityof(kern, 0.0)
end

@testset "KernelDirac Bijectors" begin
    @test maximum(KernelDirac(Float16(3))) == 3
    @test minimum(KernelDirac(Float16(3))) == 3
    @test insupport(KernelDirac(Float16(3)), 3)
    @test !insupport(KernelDirac(Float16(3)), Inf)
    @test !insupport(KernelDirac(Float16(3)), 2.9)
    @test bijector(KernelDirac(2)) == ZeroIdentity()
end

@testset "KernelDirac Transformed, RNG: $rng" for rng in rngs
    # Scalar
    d = transformed(KernelDirac(2.0))
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = transformed(KernelDirac(Float16(2)))
    x = @inferred rand(rng, d)
    @test x isa Float16
    l = @inferred logdensityof(d, x)
    @test l isa Float16

    # Array
    d = transformed(KernelDirac(2.0))
    x = @inferred rand(rng, d, 420)
    @test x isa AbstractVector{Float64}
    @test minimum(x) == 2 == maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = transformed(KernelDirac(Float16(2)))
    x = @inferred rand(rng, d, 420)
    @test x isa AbstractVector{Float16}
    @test minimum(x) == 2 == maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float16}
end

@testset "KernelDirac Transformed vs. Distributions.jl" begin
    # Compare to Distributions.jl
    dist = transformed(Dirac(3.0))
    kern = transformed(KernelDirac(3.0))

    @test logdensityof(kern, 0.9) == logdensityof(dist, 0.9)
    @test logdensityof(kern, 1.0) == logdensityof(dist, 1.0)
    @test logdensityof(kern, 1.5) == logdensityof(dist, 1.5)
    @test logdensityof(kern, 2.0) == logdensityof(dist, 2.0)
    @test logdensityof(kern, 2.1) == logdensityof(dist, 2.1)
end
