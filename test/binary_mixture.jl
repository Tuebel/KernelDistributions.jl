# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

@testset "BinaryMixture, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred BinaryMixture(KernelExponential(2.0), KernelUniform(3.0, 4.0), 3.0, 1)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = @inferred BinaryMixture(KernelExponential(Float32(2)), KernelUniform(Float32(3), Float32(4)), Float32(3), Float32(1))
    x = @inferred rand(rng, d)
    @test x isa Float32
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = @inferred BinaryMixture(KernelExponential(2.0), KernelUniform(3.0, 4.0), 3.0, 1.0)
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float64}
    @test 0 < minimum(x) < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = @inferred BinaryMixture(KernelExponential(Float32(2)), KernelUniform(Float32(3), Float32(4)), Float32(3), Float32(1))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float32}
    @test 0 < minimum(x) < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "BinaryMixture vs. Distributions.jl" begin
    dist = MixtureModel([Exponential(2.0), Uniform(3.0, 4.0)], [3.0 / 4, 1.0 / 4])
    kern = BinaryMixture(KernelExponential(2.0), KernelUniform(3.0, 4.0), 3.0, 1.0)

    @test logdensityof(dist, -1.0) == logdensityof(kern, -1.0)
    @test logdensityof(dist, 2.9) == logdensityof(kern, 2.9)
    @test logdensityof(dist, 3.0) == logdensityof(kern, 3.0)
    @test logdensityof(dist, 3.1) == logdensityof(kern, 3.1)
    @test logdensityof(dist, 3.9) == logdensityof(kern, 3.9)
    @test logdensityof(dist, 4.0) == logdensityof(kern, 4.0)
    @test logdensityof(dist, 4.1) == logdensityof(kern, 4.1)
end

@testset "BinaryMixture Bijectors" begin
    @test maximum(BinaryMixture(KernelExponential(Float32(2)), KernelUniform(Float32(3), Float32(4)), Float32(3), 1)) == Inf16
    @test minimum(BinaryMixture(KernelExponential(Float32(2)), KernelUniform(Float32(3), Float32(4)), Float32(3), 1)) == 0
    @test insupport(BinaryMixture(KernelExponential(Float32(2)), KernelUniform(Float32(3), Float32(4)), Float32(3), 1), 0)
    @test insupport(BinaryMixture(KernelExponential(Float32(2)), KernelUniform(Float32(3), Float32(4)), Float32(3), 1), Inf)
    @test !insupport(BinaryMixture(KernelExponential(Float32(2)), KernelUniform(Float32(3), Float32(4)), Float32(3), 1), -eps(Float32))
    @test bijector(BinaryMixture(KernelExponential(Float32(2)), KernelUniform(Float32(3), Float32(4)), Float32(3), 1)) == Bijectors.TruncatedBijector(0, Inf16)
end

@testset "BinaryMixture Transformed, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred transformed(BinaryMixture(KernelExponential(2.0), KernelUniform(3.0, 4.0), 3.0, 1.0))
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = @inferred transformed(BinaryMixture(KernelExponential(Float32(2)), KernelUniform(Float32(3), Float32(4)), Float32(3), Float32(1)))
    x = @inferred rand(rng, d)
    @test x isa Float32
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array
    d = @inferred transformed(BinaryMixture(KernelExponential(2.0), KernelUniform(3.0, 4.0), 3.0, 1.0))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float64}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = @inferred transformed(BinaryMixture(KernelExponential(Float32(2)), KernelUniform(Float32(3), Float32(4)), Float32(3), Float32(1)))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float32}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "BinaryMixture Transformed vs. Distributions.jl" begin
    # Compare to Distributions.jl
    dist = transformed(MixtureModel([Exponential(2.0), Uniform(3.0, 4.0)], [3.0 / 4, 1.0 / 4]))
    kern = transformed(BinaryMixture(KernelExponential(2.0), KernelUniform(3.0, 4.0), 3.0, 1.0))

    @test logdensityof(kern, 0.9) == logdensityof(dist, 0.9)
    @test logdensityof(kern, 1.0) == logdensityof(dist, 1.0)
    @test logdensityof(kern, 1.5) == logdensityof(dist, 1.5)
    @test logdensityof(kern, 2.0) == logdensityof(dist, 2.0)
    @test logdensityof(kern, 2.1) == logdensityof(dist, 2.1)
end
