# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using Distributions
using KernelDistributions
import CairoMakie as MK

"""
Width of the document in pt
"""
const DISS_WIDTH = 422.52348

change_alpha(color; alpha=0.4) = @reset color.alpha = alpha
DENSITY_PALETTE = change_alpha.(MK.Makie.wong_colors())
WONG2 = [MK.Makie.wong_colors()[4:7]..., MK.Makie.wong_colors()[1:3]...]
WONG2_ALPHA = change_alpha.(WONG2; alpha=0.2)
function wilkinson_ticks()
    wt = MK.WilkinsonTicks(5)
    @reset wt.granularity_weight = 1
end

function diss_defaults()
    # GLMakie uses the original GLAbstractions, I hijacked GLAbstractions for my purposes
    MK.set_theme!(
        palette=(; density_color=DENSITY_PALETTE, wong2=WONG2, wong2_alpha=WONG2_ALPHA),
        Axis=(; xticklabelsize=9, yticklabelsize=9, xgridstyle=:dash, ygridstyle=:dash, xgridwidth=0.5, ygridwidth=0.5, xticks=wilkinson_ticks(), yticks=wilkinson_ticks(), xticksize=0.4, yticksize=0.4, spinewidth=0.7),
        Axis3=(; xticklabelsize=9, yticklabelsize=9, zticklabelsize=9, xticksize=0.4, yticksize=0.4, zticksize=0.4, xgridwidth=0.5, ygridwidth=0.5, zgridwidth=0.5, spinewidth=0.7),
        CairoMakie=(; type="png", px_per_unit=2.0),
        Colorbar=(; width=7),
        Density=(; strokewidth=1, cycle=MK.Cycle([:color => :density_color, :strokecolor => :color], covary=true)),
        Legend=(; patchsize=(5, 5), padding=(5, 5, 5, 5), framewidth=0.7),
        Lines=(; linewidth=1),
        Scatter=(; markersize=4),
        VLines=(; cycle=[:color => :wong2], linestyle=:dash, linewidth=1),
        VSpan=(; cycle=[:color => :wong2_alpha]),
        fontsize=11, # Latex "small" for normal 12
        resolution=(DISS_WIDTH, DISS_WIDTH / 2),
        rowgap=5, colgap=5,
        figure_padding=5
    )
end

diss_defaults()

μ = 1.0
σ = 0.1
β = 1.0
z_min = 0.5
z_max = 1.5
z = 0:0.01:(z_max+0.2)
w_tail = 5
normal = KernelNormal(μ, σ)
exponential = KernelExponential(β)
smooth_exponential = SmoothExponential(0.0, μ, β, σ)
truncated_exponential = Distributions.truncated(exponential, 0.0, μ)
uniform = TailUniform()

pdf_normal = pdf.(normal, z)
pdf_exponential = pdf.(exponential, z)
pdf_smooth = pdf.(smooth_exponential, z)
pdf_truncated = pdf.(truncated_exponential, z)
pdf_uniform = 0.5 .* pdf.(uniform, z)

fig = MK.Figure(resolution=(DISS_WIDTH, 2 / 5 * DISS_WIDTH))
ax1 = MK.Axis(fig[1, 1]; title="Exponential", xlabel="depth / m", ylabel="probability density", limits=((0, nothing), (0, nothing)))
ax2 = MK.Axis(fig[1, 2]; title="Mixture", xlabel="depth / m", limits=((0, nothing), (0, nothing)))

MK.lines!(ax1, z, w_tail * pdf_uniform; label="only uniform", visible=false)
MK.lines!(ax1, z, pdf_exponential; label="unmodified exponential")
MK.lines!(ax1, z, pdf_truncated; label="truncated exponential")
MK.lines!(ax1, z, pdf_smooth; label="smooth exponential")
MK.vspan!(ax1, 0.0, μ)

MK.lines!(ax2, z, pdf_normal + w_tail * (pdf_uniform); label="only uniform")
MK.lines!(ax2, z, pdf_normal + w_tail * (pdf_exponential + pdf_uniform); label="unmodified")
MK.lines!(ax2, z, pdf_normal + w_tail * (pdf_truncated + pdf_uniform); label="truncated")
MK.lines!(ax2, z, pdf_normal + w_tail * (pdf_smooth + pdf_uniform); label="smooth")
MK.vspan!(ax2, 0.0, μ)
MK.Legend(fig[2, :], ax2; orientation=:horizontal)

MK.save("pixel_likelihoods.pdf", fig)
fig

# different classification probabilities for smooth
fig = MK.Figure(resolution=(DISS_WIDTH, 2 / 5 * DISS_WIDTH))
ax = MK.Axis(fig[1, 1]; xlabel="depth / m", ylabel="probability density", limits=((0, nothing), (0, nothing)))
for o in 0.2:0.2:0.8
    MK.lines!(ax, z, o * pdf_normal + (1 - o) * (pdf_smooth + pdf_uniform); label="o = $o")
end
MK.vspan!(ax, 0.0, μ)
MK.Legend(fig[:, 2], ax)

MK.save("pixel_likelihoods_o_smooth.pdf", fig)
fig

# different classification probabilities for exponential
fig = MK.Figure(resolution=(DISS_WIDTH, 2 / 5 * DISS_WIDTH))
ax = MK.Axis(fig[1, 1]; xlabel="depth / m", ylabel="probability density", limits=((0, nothing), (0, nothing)))
for o in 0.2:0.2:0.8
    MK.lines!(ax, z, o * pdf_normal + (1 - o) * (pdf_exponential + pdf_uniform); label="o = $o")
end
MK.vspan!(ax, 0.0, μ)
MK.Legend(fig[:, 2], ax)

MK.save("pixel_likelihoods_o_unmodified.pdf", fig)
fig

# different classification probabilities for truncated
fig = MK.Figure(resolution=(DISS_WIDTH, 2 / 5 * DISS_WIDTH))
ax = MK.Axis(fig[1, 1]; xlabel="depth / m", ylabel="probability density", limits=((0, nothing), (0, nothing)))
for o in 0.2:0.2:0.8
    MK.lines!(ax, z, o * pdf_normal + (1 - o) * (pdf_truncated + pdf_uniform); label="o = $o")
end
MK.vspan!(ax, 0.0, μ)
MK.Legend(fig[:, 2], ax)

MK.save("pixel_likelihoods_o_truncated.pdf", fig)
fig