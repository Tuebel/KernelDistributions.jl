using Plots
using Quaternions
using KernelDistributions
using Rotations

plotly()

# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
function sphere_scatter(rotations, point=[0, 0, 1]; step=5, markersize=0.5, markeralpha=0.25, kwargs...)
    r_points = map(r -> Vector(r * point), rotations)
    r_scat = hcat(r_points...)
    scatter3d(r_scat[1, begin:step:end], r_scat[2, begin:step:end], r_scat[3, begin:step:end]; markersize=markersize, markeralpha=markeralpha, kwargs...)
end

# Not uniformly distributed rotations
quats = [rand(QuaternionF64) |> sign for _ in 1:20_000]
rots = QuatRotation.(quats)
sphere_scatter(rots)

# Uniformly distributed rotations
quats = [randn(QuaternionF64) |> sign for _ in 1:20_000]
rots = QuatRotation.(quats)
sphere_scatter(rots)

# Uniformly distributed rotations
quats = [rand(QuaternionUniform()) |> sign for _ in 1:20_000]
rots = QuatRotation.(quats)
sphere_scatter(rots)