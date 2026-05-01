### WarpedArray
"""
    W = WarpedArray(A, ϕ)

Lazy array for which `W[x] = A[ϕ(x)]`, where `A` is the source array and `ϕ`
is an `AbstractDeformation`. Indexing is evaluated on demand.

`A` can be any array; it is automatically wrapped in an extrapolation object
so that out-of-bounds accesses return `NaN` rather than throwing an error.

See also [`warp`](@ref), [`warp!`](@ref), [`getindex!`](@ref).
"""
struct WarpedArray{T, N, A <: Extrapolatable, D <: AbstractDeformation} <: AbstractArray{T, N}
    data::A
    ϕ::D
end

# User already supplied an interpolatable ϕ
function WarpedArray(
        data::Extrapolatable{T, N},
        ϕ::GridDeformation{S, N, A}
    ) where {T, N, S, A <: AbstractInterpolation}
    return WarpedArray{T, N, typeof(data), typeof(ϕ)}(data, ϕ)
end

# Create an interpolatable ϕ
function WarpedArray(data::Extrapolatable{T, N}, ϕ::GridDeformation) where {T, N}
    itp = scale(interpolate(ϕ.u, BSpline(Quadratic(Flat(OnCell())))), ϕ.nodes...)
    ϕ′ = GridDeformation(itp, ϕ.nodes)
    return WarpedArray{T, N, typeof(data), typeof(ϕ′)}(data, ϕ′)
end

WarpedArray(data, ϕ::GridDeformation) = WarpedArray(to_etp(data), ϕ)


Base.size(A::WarpedArray) = size(A.data)
Base.size(A::WarpedArray, i::Integer) = size(A.data, i)
Base.axes(A::WarpedArray) = axes(A.data)
Base.axes(A::WarpedArray, i::Integer) = axes(A.data, i)

@inline function Base.getindex(W::WarpedArray{T, N}, I::Vararg{Number, N}) where {T, N}
    ϕx = W.ϕ(I...)
    return W.data(ϕx...)
end

"""
    getindex!(dest, W::WarpedArray, coords...)

Fill `dest` with values from the `WarpedArray` `W` at the Cartesian product of
`coords`. Each element of `coords` specifies the indices along one dimension.
Returns `dest`.
"""
function ImageAxes.getindex!(dest, W::WarpedArray{T, N}, coords::Vararg{Any, N}) where {T, N}
    for (i, c) in zip(LinearIndices(dest), Iterators.product(coords...))
        dest[i] = W[c...]
    end
    return dest
end
