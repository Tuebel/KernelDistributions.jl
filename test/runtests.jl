# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Bijectors
using CUDA
using KernelDistributions
using Quaternions
using Random
using SpecialFunctions
using StatsFuns
using Test

# Setup a list of rngs to loop over
cpurng = Random.default_rng()
Random.seed!(cpurng, 42)
rngs = [cpurng]
if CUDA.functional()
    curng = CUDA.default_rng()
    Random.seed!(curng, 42)
    rngs = [rngs..., curng]
end

# BUG CUDA does weird stuff (wrong calculations, minimum is always 0.0) only on my laptop during Pkg.test() not when include("runtests.jl")

include("binary_mixture.jl")
include("circular_uniform.jl")
include("dirac.jl")
include("exponential.jl")
include("normal.jl")
include("quaternion_uniform.jl")
include("smooth_exponential.jl")
include("tail_uniform.jl")
include("uniform.jl")
