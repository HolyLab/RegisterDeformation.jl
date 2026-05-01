"""
    ϕs = tinterpolate(ϕsindex, tindex, nstack)

Linearly interpolate/extrapolate a sparse set of deformations in time to fill
all frames `1:nstack`. `ϕsindex` is the vector of known deformations defined at
the integer time indices `tindex`. Satisfies `ϕs[tindex[i]] == ϕsindex[i]`.
"""
function tinterpolate(ϕsindex, tindex, nstack)
    ϕs = Vector{eltype(ϕsindex)}(undef, nstack)
    # Before the first tindex
    k = 0
    for i in 1:(tindex[1] - 1)
        ϕs[k += 1] = ϕsindex[1]
    end
    # Within tindex
    for i in 2:length(tindex)
        ϕ1 = ϕsindex[i - 1]
        ϕ2 = ϕsindex[i]
        Δt = tindex[i] - tindex[i - 1]
        for j in 0:(Δt - 1)
            α = convert(eltype(eltype(ϕ1.u)), j / Δt)
            ϕs[k += 1] = GridDeformation((1 - α) * ϕ1.u + α * ϕ2.u, ϕ2.nodes)
        end
    end
    # After the last tindex
    for i in tindex[end]:nstack
        ϕs[k += 1] = ϕsindex[end]
    end
    return ϕs
end

"""
    ϕs′ = tmedfilt(ϕs, n)

Perform temporal median-filtering on a sequence of deformations with window size `n`
(which must be odd). This is a form of smoothing that does not "round the corners" on
sudden (but persistent) shifts.

See also [`tmedfilt!`](@ref).
"""
function tmedfilt(ϕs::AbstractVector{D}, n) where {D <: AbstractDeformation}
    nhalf = n >> 1
    2nhalf + 1 == n || error("filter size must be odd")
    T = eltype(eltype(D))
    v = Array{T}(undef, ndims(D), n)
    vs = ntuple(d -> view(v, d, :), ndims(D))
    ϕ1 = copy(ϕs[1])
    ϕout = Vector{typeof(ϕ1)}(undef, length(ϕs))
    ϕout[1] = ϕ1
    return _medfilt!(ϕout, ϕs, v, vs)  # function barrier due to instability of vs
end

"""
    tmedfilt!(out, ϕs, n)

In-place version of [`tmedfilt`](@ref). Writes the filtered deformations into the
pre-allocated vector `out`, which must have the same length as `ϕs`.
"""
function tmedfilt!(out::AbstractVector, ϕs::AbstractVector{D}, n) where {D <: AbstractDeformation}
    nhalf = n >> 1
    2nhalf + 1 == n || error("filter size must be odd")
    length(out) == length(ϕs) || throw(DimensionMismatch("out and ϕs must have the same length"))
    T = eltype(eltype(D))
    v = Array{T}(undef, ndims(D), n)
    vs = ntuple(d -> view(v, d, :), ndims(D))
    out[1] = copy(ϕs[1])
    return _medfilt!(out, ϕs, v, vs)
end

@noinline function _medfilt!(ϕout, ϕs, v, vs::NTuple{N}) where {N}
    n = size(v, 2)
    nhalf = n >> 1
    tmp = Vector{eltype(eltype(vs))}(undef, N)
    u1 = ϕout[1].u
    for i in (1 + nhalf):(length(ϕs) - nhalf)
        u = similar(u1)
        for I in eachindex(ϕs[i].u)
            for j in -nhalf:nhalf
                utmp = ϕs[i + j].u[I]
                for d in 1:N
                    v[d, j + nhalf + 1] = utmp[d]
                end
            end
            for d in 1:N
                tmp[d] = median!(vs[d])
            end
            u[I] = tmp
        end
        ϕout[i] = GridDeformation(u, ϕs[i].nodes)
    end
    # Copy the beginning and end
    for i in 2:nhalf  # we did [1] in medfilt
        ϕout[i] = copy(ϕs[i])
    end
    for i in (length(ϕs) - nhalf + 1):length(ϕs)
        ϕout[i] = copy(ϕs[i])
    end
    return ϕout
end
