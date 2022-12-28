# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

@testset "KernelUniform, RNG: $rng" for rng in rngs
    # Scalar
    d = @inferred KernelUniform(2, 3.0)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = KernelUniform(Float16(2), Float16(3))
    x = @inferred rand(rng, d)
    @test x isa Float16
    l = @inferred logdensityof(d, x)
    @test l isa Float16

    # Array
    d = KernelUniform(2.0, 3.0)
    x = @inferred rand(rng, d, 4_200)
    @test 2 <= minimum(x) < maximum(x) <= 3
    @test x isa AbstractVector{Float64}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = KernelUniform(Float16(2), Float16(3))
    x = @inferred rand(rng, d, 4_200)
    @test 2 <= minimum(x) < maximum(x) <= 3
    @test x isa AbstractVector{Float16}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float16}
end

@testset "KernelUniform vs Distributions.jl" begin
    kern = KernelUniform(2.0, 3.0)
    dist = Uniform(2.0, 3.0)

    @test logdensityof(kern, -Inf) == logdensityof(dist, -Inf)
    @test logdensityof(kern, 1.9) == logdensityof(dist, 1.9)
    @test logdensityof(kern, 2.0) == logdensityof(dist, 2.0)
    @test logdensityof(kern, 2.1) == logdensityof(dist, 2.1)
    @test logdensityof(kern, 2.9) == logdensityof(dist, 2.9)
    @test logdensityof(kern, 3.0) == logdensityof(dist, 3.0)
    @test logdensityof(kern, 3.1) == logdensityof(dist, 3.1)
    @test logdensityof(kern, Inf) == logdensityof(dist, Inf)
    @test logcdf(kern, 1.9) == logcdf(dist, 1.9)
    @test logcdf(kern, 2.0) == logcdf(dist, 2.0)
    @test logcdf(kern, 2.1) == logcdf(dist, 2.1)
    @test logcdf(kern, 2.9) == logcdf(dist, 2.9)
    @test logcdf(kern, 3.0) == logcdf(dist, 3.0)
    @test logcdf(kern, 3.1) == logcdf(dist, 3.1)
    @test logcdf(kern, Inf) == logcdf(dist, Inf)
end

@testset "KernelUniform Bijectors" begin
    @test maximum(KernelUniform(Float16)) == 1
    @test minimum(KernelUniform(Float16)) == 0
    @test insupport(KernelUniform(Float16), 0)
    @test insupport(KernelUniform(Float16), 1)
    @test !insupport(KernelUniform(Float16), -0.1)
    @test !insupport(KernelUniform(Float16), -1.1)
    @test bijector(KernelUniform(2.0, 3.0)) == bijector(Uniform(2.0, 3.0))
end

@testset "KernelUniform Transformed, RNG: $rng" for rng in rngs
    # Scalar
    d = transformed(KernelUniform(2.0, 3.0))
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = transformed(KernelUniform(Float16(2), Float16(3)))
    x = @inferred rand(rng, d)
    @test x isa Float16
    l = @inferred logdensityof(d, x)
    @test l isa Float16

    # Array
    d = transformed(KernelUniform(2.0, 3.0))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float64}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    # Float16 fails due to precision
    d = transformed(KernelUniform(Float16(2), Float16(3)))
    x = @inferred rand(rng, d, 4_200)
    @test x isa AbstractVector{Float16}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float16}
end

@testset "KernelUniform Transformed vs. Distributions.jl" begin
    # Compare to Distributions.jl
    dist = transformed(Uniform(2.0, 3.0))
    kern = transformed(KernelUniform(2.0, 3.0))

    @test logdensityof(kern, -Inf) == logdensityof(dist, -Inf)
    @test logdensityof(kern, 1.9) == logdensityof(dist, 1.9)
    @test logdensityof(kern, 2.0) == logdensityof(dist, 2.0)
    @test logdensityof(kern, 2.1) == logdensityof(dist, 2.1)
    @test logdensityof(kern, 2.9) == logdensityof(dist, 2.9)
    @test logdensityof(kern, 3.0) == logdensityof(dist, 3.0)
    @test logdensityof(kern, 3.1) == logdensityof(dist, 3.1)
    @test logdensityof(kern, Inf) == logdensityof(dist, Inf)
end
