using CoordinateTransformations: SphericalFromCartesian
using CairoMakie
using IterTools: partition
using KernelDistributions
using LinearAlgebra
using Quaternions
using Rotations
using StatsBase: fit, Histogram

# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 
function sphere_scatter(rotations, point=[0, 0, 1]; step=5, markersize=0.5, markeralpha=0.25, kwargs...)
    r_points = map(r -> Vector(r * point), rotations)
    r_scat = hcat(r_points...)
    scatter3d(r_scat[1, begin:step:end], r_scat[2, begin:step:end], r_scat[3, begin:step:end]; markersize=markersize, markeralpha=markeralpha, kwargs...)
end

"""
    sphere_density(rotations, [point]; [n_θ, n_ϕ, color], kwargs...)
Plot the density of the rotations by rotating a point on the unit sphere.
The density of the rotations is visualized as a heatmap and takes into account the non-uniformity of the patches' surface area on a sphere.
"""
function sphere_density(rotations, point=[0, 0, 1]; n_θ=50, n_ϕ=25, color=:viridis, kwargs...)
    # rotate on unit sphere
    r_points = map(r -> Vector(r * normalize(point)), rotations)
    # histogram of spherical coordinates: θ ∈ [-π,π], ϕ ∈ [-π/2,π/2]
    spherical = SphericalFromCartesian().(r_points)
    θ_hist = range(-π, π; length=n_θ + 1)
    ϕ_hist = range(-π / 2, π / 2; length=n_ϕ + 1)
    hist = fit(Histogram, (getproperty.(spherical, :θ), getproperty.(spherical, :ϕ)), (θ_hist, ϕ_hist))

    # sphere surface patches do not have a uniform area, calculate actual patch area using the sphere  integral for r=1
    ∫_unitsphere(θ_l, θ_u, ϕ_l, ϕ_u) = (cos(ϕ_l) - cos(ϕ_u)) * (θ_u - θ_l)
    # ∫_unitsphere(θ_l, θ_u, ϕ_l, ϕ_u) = (cos(θ_l) - cos(θ_u)) * (ϕ_u - ϕ_l)
    # different range than the spherical coordinates conversion above
    θ_patch = range(-π, π; length=n_θ + 1)
    ϕ_patch = range(0, π; length=n_ϕ + 1)
    patches = [∫_unitsphere(θ..., ϕ...) for θ in partition(θ_patch, 2, 1), ϕ in partition(ϕ_patch, 2, 1)]
    # area correction & max-norm
    weights = normalize(hist.weights ./ patches, Inf)
    # weights = hist.weights ./ patches

    # parametrize surface for the plot
    θ_surf = range(-π, π; length=n_θ)
    ϕ_surf = range(0, π; length=n_ϕ)
    x_surf = cos.(θ_surf) * sin.(ϕ_surf)'
    y_surf = sin.(θ_surf) * sin.(ϕ_surf)'
    z_surf = ones(n_θ) * cos.(ϕ_surf)'

    # override fill_z to use the weights for the surface color
    surface(x_surf, y_surf, z_surf; color=weights)
end


# Not uniformly distributed rotations
eulers = [RotZYX((2π * rand(3))...) for _ in 1:500_000]
# Reduce number of patches for reasonable PDF sizes
scene = sphere_density(eulers; n_θ=15, n_ϕ=15)
save("plot_euler.pdf", scene)

# Not uniformly distributed rotations
quats = [rand(QuaternionF64) |> sign for _ in 1:500_000]
rots = QuatRotation.(quats)
scene = sphere_density(rots; n_θ=15, n_ϕ=15)
save("plot_quat.pdf", scene)

# Uniformly distributed rotations
quats = [randn(QuaternionF64) |> sign for _ in 1:500_000]
rots = QuatRotation.(quats)
scene = sphere_density(rots; n_θ=15, n_ϕ=15)
save("plot_uniform.pdf", scene)

# Uniformly distributed rotations
quats = [rand(QuaternionUniform()) |> sign for _ in 1:500_000]
rots = QuatRotation.(quats)
scene = sphere_density(rots; n_θ=15, n_ϕ=15)
