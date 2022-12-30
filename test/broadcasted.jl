# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

@test BroadcastedDistribution(KernelExponential, 2.0) |> show |> isnothing

@testset "Product BroadcastedDistribution, RNG: $rng" for rng in rngs
    # Scalar single
    d = @inferred BroadcastedDistribution(KernelExponential, 2.0)
    x = @inferred rand(rng, d)
    @test x isa Float64
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = @inferred BroadcastedDistribution(KernelExponential, 2.0f0)
    x = @inferred rand(rng, d)
    @test x isa Float32
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Scalar multiple
    d = @inferred BroadcastedDistribution(KernelExponential, 2.0)
    x = @inferred rand(rng, d, 42)
    @test x isa AbstractVector{Float64}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = @inferred BroadcastedDistribution(KernelExponential, 2.0f0)
    x = @inferred rand(rng, d, 42)
    @test x isa AbstractVector{Float32}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}

    # Array single
    d = @inferred BroadcastedDistribution(KernelExponential, rand(rng, Float64, 42))
    x = @inferred rand(rng, d)
    @test x isa AbstractVector{Float64}
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    d = @inferred BroadcastedDistribution(KernelExponential, rand(rng, Float32, 42, 42))
    x = @inferred rand(rng, d)
    @test x isa AbstractMatrix{Float32}
    l = @inferred logdensityof(d, x)
    @test l isa Float32

    # Array multiple
    d = @inferred BroadcastedDistribution(KernelExponential, rand(rng, Float64, 42))
    x = @inferred rand(rng, d, 42)
    @test x isa AbstractMatrix{Float64}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = @inferred BroadcastedDistribution(KernelExponential, rand(rng, Float32, 42, 42))
    x = @inferred rand(rng, d, 42)
    @test x isa AbstractArray{Float32,3}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float32}
end

@testset "Product BroadcastedDistribution vs. Distributions.jl" begin
    params = [0.5, 1.0, 2.0]
    dist = Product(Exponential.(params))
    kern = BroadcastedDistribution(KernelExponential, params)

    @test logdensityof(dist, [0.0, 0.0, 0.0]) == logdensityof(kern, [0.0, 0.0, 0.0])
    @test logdensityof(dist, [1.0, 1.0, 1.0]) == logdensityof(kern, [1.0, 1.0, 1.0])
end

@testset "BroadcastedDistribution dims, RNG: $rng" for rng in rngs
    # Array single
    d = @inferred BroadcastedDistribution(KernelExponential, (), rand(rng, Float64, 42))
    x = @inferred rand(rng, d)
    @test x isa AbstractVector{Float64}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = @inferred BroadcastedDistribution(KernelExponential, (), rand(rng, Float32, 42, 42))
    x = @inferred rand(rng, d)
    @test x isa AbstractMatrix{Float32}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractMatrix{Float32}

    # Array multiple
    d = @inferred BroadcastedDistribution(KernelExponential, (2,), rand(rng, Float64, 42))
    x = @inferred rand(rng, d, 24)
    @test x isa AbstractMatrix{Float64}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}
    @test size(l) == (42,)

    d = @inferred BroadcastedDistribution(KernelExponential, (2,), rand(rng, Float32, 24, 42))
    x = @inferred rand(rng, d, 11)
    @test x isa AbstractArray{Float32,3}
    l = @inferred logdensityof(d, x)
    @test l isa AbstractMatrix{Float32}
    @test size(l) == (24, 11)
end

@testset "BroadcastedDistribution Truncated, RNG: $rng" for rng in rngs
    # Scalar single
    params = 2 .* one.(array_for_rng(rng, Float64, 420))
    d = @inferred truncated(BroadcastedDistribution(KernelExponential, params), 1.0, 2.0)
    x = @inferred rand(rng, d)
    @test x isa AbstractVector{Float64}
    @test 1 <= minimum(x) < maximum(x) <= 2
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    # Scalar multiple
    x = @inferred rand(rng, d, 24)
    @test x isa AbstractMatrix{Float64}
    @test 1 <= minimum(x) < maximum(x) <= 2
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    # Array single
    params = 2 .* one.(array_for_rng(rng, Float64, 420))
    lower = 0.5 * params
    upper = params
    d = @inferred truncated(BroadcastedDistribution(KernelExponential, params), lower, upper)
    x = @inferred rand(rng, d)
    @test x isa AbstractVector{Float64}
    @test 1 <= minimum(x) < maximum(x) <= 2
    l = @inferred logdensityof(d, x)
    @test l isa Float64

    # Array multiple
    x = @inferred rand(rng, d, 24)
    @test x isa AbstractMatrix{Float64}
    @test 1 <= minimum(x) < maximum(x) <= 2
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}
end

@testset "BroadcastedDistribution Bijectors" begin
    d = BroadcastedDistribution(KernelExponential, 1.0)
    b = bijector(d)
    @test b isa BroadcastedBijector{0}

    d = BroadcastedDistribution(KernelExponential, [0.5, 1.0, 2.0])
    b = bijector(d)
    @test b isa BroadcastedBijector{1}
end

@testset "BroadcastedDistribution Transformed, RNG: $rng" for rng in rngs
    # Array single
    d = @inferred transformed(BroadcastedDistribution(KernelExponential, (), rand(rng, Float64, 420)))
    x = @inferred rand(rng, d)
    @test x isa AbstractVector{Float64}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}

    d = @inferred transformed(BroadcastedDistribution(KernelExponential, (), rand(rng, Float32, 42, 42)))
    x = @inferred rand(rng, d)
    @test x isa AbstractMatrix{Float32}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractMatrix{Float32}

    # Array multiple
    d = @inferred transformed(BroadcastedDistribution(KernelExponential, (2,), rand(rng, Float64, 42)))
    x = @inferred rand(rng, d, 24)
    @test x isa AbstractMatrix{Float64}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractVector{Float64}
    @test size(l) == (42,)

    d = @inferred transformed(BroadcastedDistribution(KernelExponential, (2,), rand(rng, Float32, 24, 42)))
    x = @inferred rand(rng, d, 11)
    @test x isa AbstractArray{Float32,3}
    @test minimum(x) < 0 < maximum(x)
    l = @inferred logdensityof(d, x)
    @test l isa AbstractMatrix{Float32}
    @test size(l) == (24, 11)
end

@testset "BroadcastedDistribution Transformed vs. KernelExponential, RNG: $rng" for rng in rngs
    one_a = one.(array_for_rng(rng, Float64, 42))
    params = 3 .* one_a
    # Compare to Distributions.jl
    dist = transformed(KernelExponential(3.0))
    kern = transformed(BroadcastedDistribution(KernelExponential, (), params))

    @test logdensityof(kern, 0 .* one_a) == logdensityof(dist, 0 .* one_a)
    @test logdensityof(kern, one_a) == logdensityof(dist, one_a)
    @test logdensityof(kern, 100 .* one_a) == logdensityof(dist, 100 .* one_a)
end
