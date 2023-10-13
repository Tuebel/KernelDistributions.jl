# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Bijectors
using CUDA
using KernelDistributions
using LinearAlgebra
using Random
using SpecialFunctions
using StatsFuns
using Test

# Setup a list of rngs to loop over
cpurng = Random.default_rng()
Random.seed!(cpurng, 42)
curng = CUDA.default_rng()
Random.seed!(curng, 42)
rngs = [cpurng, curng]

CUDA.allowscalar(false)

include("circular_uniform.jl")
include("dirac.jl")
include("exponential.jl")
include("normal.jl")
include("quaternion_perturbation.jl")
include("quaternion_uniform.jl")
include("smooth_exponential.jl")
include("tail_uniform.jl")
include("uniform.jl")
# Base distributions should work before it is worth debugging the compositions
include("binary_mixture.jl")
include("broadcasted.jl")
