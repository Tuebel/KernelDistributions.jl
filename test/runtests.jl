# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Bijectors
using CUDA
using KernelDistributions
using Random
using Test

# Setup a list of rngs to loop over
cpurng = Random.default_rng()
Random.seed!(cpurng, 42)
rngs = [cpurng]
if CUDA.functional()
    curng = CUDA.default_rng()
    Random.seed!(42)
    rngs = [rngs..., curng]
end


include("kernel_exponential.jl")

