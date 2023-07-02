# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using KernelDistributions
using LinearAlgebra
using Quaternions
using Random
using Test

σ = 0.01

"""
    exponential_map(x, y, z)
Convert a rotation vector to a Quaternion (formerly qrotation in Quaternions.jl)
Eq. (101) in J. Sola, „Quaternion kinematics for the error-state KF“
Exponential for quaternions can be reformulated to the exponential map using (46).
"""
function exponential_map(x, y, z)
    rotvec = [x, y, z]
    theta = norm(rotvec)
    s, c = sincos(theta / 2)
    scaleby = s / (iszero(theta) ? one(theta) : theta)
    Quaternion(c, scaleby * rotvec[1], scaleby * rotvec[2], scaleby * rotvec[3])
end

# (eq. 105, Sola2012)
function logarithmic_map(q)
    qv = [q.v1, q.v2, q.v3]
    abs_qv = norm(qv)
    ϕ = 2 * atan(abs_qv, q.s)
    u = qv / abs_qv
    ϕ * u
end

@testset "Additive and subtractive quaternion operators" begin
    # Normalization approximation
    θ = rand(KernelNormal(0, Float32(σ)), 3)
    q = @inferred KernelDistributions.exp_map(θ)
    @test abs(q) == 1
    @test q isa QuaternionF32
    @test !isone(q)
    @test q ≈ exponential_map(θ...)
    @test θ ≈ KernelDistributions.log_map(q) ≈ logarithmic_map(q)

    # add rotation to quaternion
    qs = @inferred one(QuaternionF32) ⊕ θ
    @test qs isa QuaternionF32
    @test qs == q
    # broadcastable?
    Qs1 = @inferred one(QuaternionF32) .⊕ fill(θ, 42)
    @test reduce(&, Qs1 .== qs)
    @test Qs1 isa Vector{QuaternionF32}
    @test length(Qs1) == 42
    # [θ] to broadcast along first dimension
    Qs2 = fill(one(QuaternionF32), 42) .⊕ [θ]
    @test Qs1 == Qs2

    # subtract rotation from quaternion
    qs = @inferred one(QuaternionF32) ⊖ θ
    @test qs isa QuaternionF32
    @test qs == Quaternion(real(q), -1 .* imag_part(q)...)
    @test abs(qs) == 1
    # broadcastable?
    Qs1 = @inferred one(QuaternionF32) .⊖ fill(θ, 42)
    @test reduce(&, Qs1 .== qs)
    @test Qs1 isa Vector{QuaternionF32}
    @test length(Qs1) == 42
    # [θ] to broadcast along first dimension
    Qs2 = fill(one(QuaternionF32), 42) .⊖ [θ]
    @test Qs1 == Qs2

    # subtract quaternion from quaternion
    qs = @inferred one(QuaternionF32) ⊕ θ
    @test qs ⊖ one(QuaternionF32) ≈ θ
    q = randn(QuaternionF32)
    qs = q ⊕ θ
    @test θ ≈ qs ⊖ q
    @test qs ⊖ q ≈ -(q ⊖ qs)
    @test (qs ⊖ q) isa Vector{Float32}
    # broadcastable?
    Qs = @inferred q .⊕ fill(θ, 42)
    Θ = @inferred Qs .⊖ q
    @test reduce(&, Θ .≈ [θ])
    @test Θ isa Vector{Vector{Float32}}
    @test length(Θ) == 42
end
