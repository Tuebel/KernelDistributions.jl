# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2023, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Accessors
using Distributions
using KernelDistributions
using Random
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

rng = Random.default_rng()
Random.seed!(rng, 2474541)
diss_defaults()
d = KernelNormal(1.0, 1.0)
y = rand(rng, d, 10_000)
x = exp.(y)

fig = MK.Figure(resolution=(DISS_WIDTH, 1 / 4 * DISS_WIDTH))
ax1 = MK.Axis(fig[1, 1]; title="y ~ 𝓝 (1,1)", ylabel="probability density")
ax2 = MK.Axis(fig[1, 2]; title="x=b⁻¹(y)=exp(y)")
MK.density!(ax1, y)
MK.density!(ax2, x)
MK.save("transform_density.pdf", fig)
fig
