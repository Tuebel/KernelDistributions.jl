# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# Broadcast bijector for array of distributions since each might have different parameters
Bijectors.bijector(dists::AbstractArray{<:KernelOrTransformedKernel}) = bijector.(dists)

# By default, Distributions.jl disallows logdensityof with multiple samples (Arrays and Matrices). KernelDistributions should be inherently allowing multiple samples.
DensityInterface.logdensityof(dist::KernelOrKernelArray, x::AbstractArray) = logpdf.(dist, x)

"""
    array_for_rng(rng, T, dims...)
Generate the correct array to be used in rand! based on the random number generator provided.
CuArray for CUDA.RNG and Array for all other RNGs.
"""
array_for_rng(rng::AbstractRNG, ::Type{T}, dims::Integer...) where {T} = array_for_rng(rng){T}(undef, dims...)
array_for_rng(::AbstractRNG) = Array
array_for_rng(::CUDA.RNG) = CuArray

# Make KernelDistributions extendable by allowing to override it for custom distributions
array_for_rng(rng::AbstractRNG, ::KernelOrTransformedKernel{T}, dims::Integer...) where {T} = array_for_rng(rng, T, dims...)
array_for_rng(rng::AbstractRNG, array::AbstractArray{<:KernelOrTransformedKernel{T}}, dims::Integer...) where {T} = array_for_rng(rng, T, size(array)..., dims...)

"""
    rand(rng, dist, [dims...])
Sample an Array from `dist` of size `dims`.
"""
function Base.rand(rng::AbstractRNG, dist::KernelOrTransformedKernel, dims::Int64...)
    A = array_for_rng(rng, dist, dims...)
    rand!(rng, dist, A)
end


"""
    rand(rng, dist, [dims...])
Sample an Array from `dists` of size `dims`.
"""
function Base.rand(rng::AbstractRNG, dists::AbstractArray{<:KernelOrTransformedKernel}, dims::Integer...)
    A = array_for_rng(rng, dists, dims...)
    rand!(rng, dists, A)
end

"""
    rand!(rng, dist, A)
Mutate the array A by sampling from `dist`.
"""
Random.rand!(rng::AbstractRNG, dist::AbstractArray{<:KernelOrTransformedKernel}, A::AbstractArray) = rand_kernel!(rng, dist, A)
Random.rand!(rng::AbstractRNG, dist::KernelOrTransformedKernel, A::AbstractArray) = rand_kernel!(rng, dist, A)
# Resolve ambiguities with Distributions.jl
Random.rand!(rng::AbstractRNG, dist::KernelOrTransformedKernel, A::AbstractArray{<:Real}) = rand_kernel!(rng, dist, A)

# CPU implementation

"""
    kernel_rand!(rng, dist, A)
Internal inplace random function which allows dispatching based on the RNG and output array.
Keeping dist as Any allows more flexibility, for example passing a Broadcasted to avoid allocations.
"""
function rand_kernel!(rng::AbstractRNG, dist, A::AbstractArray)
    # Avoid endless recursions for rand(rng, dist::KernelOrTransformedKernel)
    @. A = rand_kernel(rng, dist)
end

# GPU implementation

# Currently only the CUDA.RNG is supported in Kernels.
function rand_kernel!(rng::CUDA.RNG, dist, A::CuArray)
    rand_cuda_kernel(dist, A, rng.seed, rng.counter)
    new_counter = Int64(rng.counter) + length(A)
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow
    rng.counter = remainder
    return A
end

# Function barrier for CUDA.RNG which is not isbits.
# Wrapping rng in Tuple for broadcasting does not work → anonymous function is the workaround 
# Thanks vchuravy https://github.com/JuliaGPU/CUDA.jl/issues/1480#issuecomment-1102245813
function rand_cuda_kernel(dist, A, seed, counter)
    A .= (x -> rand_kernel(device_rng(seed, counter), x)).(dist)
end

"""
    device_rng(seed, counter)
Use it inside a kernel to generate a correctly seeded device RNG.
"""
function device_rng(seed, counter)
    # Replaced during kernel compilation: https://github.com/JuliaGPU/CUDA.jl/blob/778f7fa21f3f73841a2dada57767e358f80e5997/src/device/random.jl#L37
    rng = Random.default_rng()
    # Same as in CUDA.jl: https://github.com/JuliaGPU/CUDA.jl/blob/778f7fa21f3f73841a2dada57767e358f80e5997/src/random.jl#L79
    @inbounds Random.seed!(rng, seed, counter)
    rng
end

"""
    sum_and_dropdims(A, dims)
Sum the matrix A over the given dimensions and drop the very same dimensions afterwards.
In case of a matching number of dimensions, a scalar is returned
"""
# Cannot dispatch on named parameter so implement helper methods below
sum_and_dropdims(A, dims::Dims) = dropdims(sum(A; dims=dims), dims=dims)
# Case of matching dimensions → return scalar
sum_and_dropdims(A::AbstractArray{<:Any,N}, ::Dims{N}) where {N} = sum(A)
# Scalar case
sum_and_dropdims(A::Number, ::Dims{N}) where {N} = A

# WARN Do not try to implement reduction of Broadcasted via Base.mapreducedim!
# LinearIndices(::Broadcasted{<:Any,<:Tuple{Any}}) only works for 1D case: https://github.com/JuliaLang/julia/blob/v1.8.0/base/broadcast.jl#L245
# Type hijacking does not work, since Broadcasted handles indexing differently which results to different results
# Base.LinearIndices(bc::Broadcast.Broadcasted{<:Any,<:Tuple}) = LinearIndices(axes(bc))
# Base.has_fast_linear_indexing(bc::Broadcast.Broadcasted{<:Broadcast.BroadcastStyle,<:Tuple}) = false
