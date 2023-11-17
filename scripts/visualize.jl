using Accessors
using CoordinateTransformations: SphericalFromCartesian
using CairoMakie
using IterTools: partition
using KernelDistributions
using LinearAlgebra
using Quaternions
using Rotations
using StatsBase: fit, Histogram

const DISS_WIDTH = 422.52348

function wilkinson_ticks()
    wt = WilkinsonTicks(5)
    @reset wt.granularity_weight = 1
end
function diss_defaults()
    # GLMakie uses the original GLAbstractions, I hijacked GLAbstractions for my purposes
    set_theme!(
        Axis=(; xticklabelsize=9, yticklabelsize=9, xgridstyle=:dash, ygridstyle=:dash, xgridwidth=0.5, ygridwidth=0.5, xticks=wilkinson_ticks(), yticks=wilkinson_ticks(), xticksize=0.4, yticksize=0.4, spinewidth=0.7),
        Axis3=(; xticklabelsize=9, yticklabelsize=9, zticklabelsize=9, xticksize=0.4, yticksize=0.4, zticksize=0.4, xgridwidth=0.5, ygridwidth=0.5, zgridwidth=0.5, spinewidth=0.7),
        CairoMakie=(; type="png", px_per_unit=2.0),
        Colorbar=(; width=7),
        Legend=(; patchsize=(5, 5), padding=(5, 5, 5, 5), framewidth=0.7),
        Lines=(; linewidth=1),
        Scatter=(; markersize=4),
        VLines=(; cycle=[:color => :wong2], linestyle=:dash),
        VSpan=(; cycle=[:color => :wong2_alpha]),
        fontsize=11, # Latex "small" for normal 12
        resolution=(DISS_WIDTH, DISS_WIDTH / 2),
        rowgap=5, colgap=5,
        figure_padding=5
    )
end
diss_defaults()

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
function sphere_density!(axis, rotations, point=[0, 0, 1]; n_θ=50, n_ϕ=25, rasterize=3, kwargs...)
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

    # set surface color to the weights
    surface!(axis, x_surf, y_surf, z_surf; color=weights, rasterize=rasterize, kwargs...)
end

# Not uniformly distributed rotations
fig = Figure()
ax_eul = Axis3(fig[1, 1]; aspect=:equal, azimuth=45, xticks=[-1, 0, 1], yticks=[-1, 0, 1], zticks=[0, 1], title="ZYX-Euler")
ax_quat = Axis3(fig[1, 2]; aspect=:equal, azimuth=45, xticks=[-1, 0, 1], yticks=[-1, 0, 1], zticks=[0, 1], title="Quaternion")

eulers = [RotZYX((2π * rand(3))...) for _ in 1:500_000]
# Reduce number of patches for reasonable PDF sizes
sphere_density!(ax_eul, eulers; n_θ=50, n_ϕ=50)

# Uniformly distributed rotations
quats = [rand(QuaternionUniform()) |> sign for _ in 1:500_000]
rots = QuatRotation.(quats)
sphere_density!(ax_quat, rots; n_θ=50, n_ϕ=50)

Colorbar(fig[:, end+1]; size=10, label="density / -")
display(fig)
save("random_rotation.pdf", fig)

# Next plot
fig = Figure()
ax1 = Axis3(fig[1, 1]; aspect=:equal, azimuth=45)
ax2 = Axis3(fig[1, 2]; aspect=:equal, azimuth=45)

# Not uniformly distributed rotations
quats = [rand(QuaternionF64) |> sign for _ in 1:500_000]
rots = QuatRotation.(quats)
sphere_density!(ax1, rots; n_θ=20, n_ϕ=20)

# Uniformly distributed rotations
quats = [randn(QuaternionF64) |> sign for _ in 1:500_000]
rots = QuatRotation.(quats)
sphere_density!(ax2, rots; n_θ=20, n_ϕ=20)
Colorbar(fig[:, end+1])
fig
