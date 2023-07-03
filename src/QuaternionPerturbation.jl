# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
(Sola 2012): J. Sola, „Quaternion kinematics for the error-state KF“, Laboratoire dAnalyse et dArchitecture des Systemes-Centre national de la recherche scientifique (LAAS-CNRS), Toulouse, France, Tech. Rep, 2012.
"""

"""
    exp_map(x, y, z)
Convert an axis-angle rotation vector to a Quaternion (formerly qrotation in Quaternions.jl, Sola 2012)).
Exponential for quaternions can be reformulated to the exponential map using (eq. 46, Sola 2012)).
"""
exp_map(x, y, z) = exp(Quaternion(0, x / 2, y / 2, z / 2)) |> nonzero_sign # pretty much as fast as the approx version in (193)
exp_map(v) = exp_map(v...)

"""
    log_map(q)
Convert a Quaternion to an axis angle rotation vector.
Exponential for quaternions can be reformulated to the exponential map using (eq. 46, Sola 2012) - log similar.
"""
function log_map(q::Quaternion)
    log_q = log(q)
    2 .* [log_q.v1, log_q.v2, log_q.v3]
end

"""
    ⊕(q, θ)
'The ‘plus’ operator [qs = qr ⊕ θ] : SO(3) × R3 → SO(3) produces an element S of SO(3) which is the result of composing a reference element R of SO(3) with a (often small) rotation.
This rotation is specified by a vector of θ ∈ R3 in the vector space tangent [...]' (eq. 158, Sola 2012)
"""
⊕(q::Quaternion, θ) = q * exp_map(θ)
# Default to addition
⊕(a, b) = a + b

"""
    ⊖(qs, qr)
'The ‘minus’ operator [θ = qs ⊖ qr] : SO(3) × SO(3) → R3 is the inverse of [qs = qr ⊕ θ].
It returns the vectorial angular difference θ ∈ R3 between two elements of SO(3).
This difference is expressed in the vector space tangent to the reference element [...]' (eq. 161, Sola 2012)
"""
⊖(qs::Quaternion, qr::Quaternion) = log_map(qr \ qs)
# Default to subtraction
⊖(a, b) = a - b

# Broadcasting behavior of ⊕ for quaternions since the array types would result in dimensions mismatches. Alternative would be to introduce a QuaternionPerturbation type to wrap the arrays.
Broadcast.broadcasted(::typeof(⊕), q::Quaternion, v::AbstractVector{<:Real}) = Broadcast.broadcasted(⊕, Ref(q), Ref(v))
Broadcast.broadcasted(::typeof(⊕), q::Quaternion, M::AbstractMatrix{<:Real}) = Broadcast.broadcasted(⊕, Ref(q), [v for v in eachcol(M)])

Broadcast.broadcasted(::typeof(⊕), q::AbstractVector{<:Quaternion}, v::AbstractVector{<:Real}) = Broadcast.broadcasted(⊕, q, Ref(v))
Broadcast.broadcasted(::typeof(⊕), q::AbstractVector{<:Quaternion}, M::AbstractMatrix{<:Real}) = Broadcast.broadcasted(⊕, q, [v for v in eachcol(M)])
