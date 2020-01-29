### WarpedArray
"""
A `WarpedArray` `W` is an AbstractArray for which `W[x] = A[ϕ(x)]` for
some parent array `A` and some deformation `ϕ`.  The object is created
lazily, meaning that computation of the displaced values occurs only
when you ask for them explicitly.

Create a `WarpedArray` like this:

```
W = WarpedArray(A, ϕ)
```
where

- The first argument `A` is an `AbstractExtrapolation` that can be
  evaluated anywhere.  See the Interpolations package.
- ϕ is an `AbstractDeformation`
"""
struct WarpedArray{T,N,A<:Extrapolatable,D<:AbstractDeformation} <: AbstractArray{T,N}
    data::A
    ϕ::D
end

# User already supplied an interpolatable ϕ
function WarpedArray(data::Extrapolatable{T,N},
                     ϕ::GridDeformation{S,N,A}) where {T,N,S,A<:AbstractInterpolation}
    WarpedArray{T,N,typeof(data),typeof(ϕ)}(data, ϕ)
end

# Create an interpolatable ϕ
function WarpedArray(data::Extrapolatable{T,N}, ϕ::GridDeformation) where {T,N}
    itp = scale(interpolate(ϕ.u, BSpline(Quadratic(Flat(OnCell())))), ϕ.nodes...)
    ϕ′ = GridDeformation(itp, ϕ.nodes)
    WarpedArray{T,N,typeof(data),typeof(ϕ′)}(data, ϕ′)
end

WarpedArray(data, ϕ::GridDeformation) = WarpedArray(to_etp(data), ϕ)


Base.size(A::WarpedArray) = size(A.data)
Base.size(A::WarpedArray, i::Integer) = size(A.data, i)
Base.axes(A::WarpedArray) = axes(A.data)
Base.axes(A::WarpedArray, i::Integer) = axes(A.data, i)

@inline function Base.getindex(W::WarpedArray{T,N}, I::Vararg{Number,N}) where {T,N}
    ϕx = W.ϕ(I...)
    return W.data(ϕx...)
end

ImageAxes.getindex!(dest, W::WarpedArray{T,N}, coords::Vararg{Any,N}) where {T,N} =
    Base._unsafe_getindex!(dest, W, coords...)
