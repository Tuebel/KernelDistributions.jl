# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# Replace RNG for scalars to sample on CPU instead of GPU

Base.rand(rng::AbstractRNG, dist::KernelOrTransformedKernel) = rand_kernel(rng, dist)
function Base.rand(rng::CUDA.RNG, dist::KernelOrTransformedKernel)
    # Setup CPU version of the Philox2x32 used on the GPU. Convenience constructors are not type stable.
    cpu_rng = Philox2x{UInt32,7}(0, 0, 0, 0, 0, 0)
    Random.seed!(cpu_rng, rng.seed)
    set_counter!(cpu_rng, rng.counter)
    # Update the CUDA rng
    new_counter = Int64(rng.counter) + 1
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow
    rng.counter = remainder
    # Finally generate the random number
    rand_kernel(cpu_rng, dist)
end
