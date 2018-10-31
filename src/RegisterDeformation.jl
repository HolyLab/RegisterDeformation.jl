module RegisterDeformation

using Images, Interpolations, ColorTypes, StaticArrays, HDF5, JLD2, ProgressMeter
using RegisterUtilities, LinearAlgebra, Base.Cartesian
using Base: tail
import Interpolations: AbstractInterpolation, AbstractExtrapolation
import ImageTransformations: warp, warp!
# to avoid `scale` conflict:
import CoordinateTransformations: compose, AffineMap
include("tformedarrays.jl")

export
    # types
    AbstractDeformation,
    GridDeformation,
    WarpedArray,
    # functions
    arraysize,
    compose,
    eachknot,
    griddeformations,
    knotgrid,
    medfilt,
    regrid,
    tform2deformation,
    tinterpolate,
    translate,
    vecindex,
    vecgradient!,
    warp,
    warp!,
    warpgrid,
    tformtranslate,
    tformrotate

const DimsLike = Union{Vector{Int}, Dims}
const InterpExtrap = Union{AbstractInterpolation,AbstractExtrapolation}
const Extrapolatable{T,N} = Union{TransformedArray{T,N},AbstractExtrapolation{T,N}}

"""
# RegisterDeformation

A deformation (or warp) of space is represented by a function `ϕ(x)`.
For an image, the warped version of the image is specified by "looking
up" the pixel value at a location `ϕ(x) = x + u(x)`.  `u(x)` thus
expresses the displacement, in pixels, at position `x`.  Note that a
constant deformation, `u(x) = x0`, corresponds to a shift of the
*coordinates* by `x0`, and therefore a shift of the *image* in the
opposite direction.

In reality, deformations will be represented on a grid, and
interpolation is implied at locations between grid points. For a
deformation defined directly from an array, make it interpolating
using `ϕi = interpolate(ϕ)`.

The major functions/types exported by RegisterDeformation are:

    - `GridDeformation`: create a deformation
    - `tform2deformation`: convert an `AffineMap` to a deformation
    - `ϕ_old(ϕ_new)` and `compose`: composition of two deformations
    - `warp` and `warp!`: deform an image
    - `WarpedArray`: create a deformed array lazily
    - `warpgrid`: visualize a deformation

"""
RegisterDeformation

abstract type AbstractDeformation{T,N} end
Base.eltype(::Type{AbstractDeformation{T,N}}) where {T,N} = T
Base.ndims(::Type{AbstractDeformation{T,N}}) where {T,N} = N
Base.eltype(::Type{D}) where {D<:AbstractDeformation} = eltype(supertype(D))
Base.ndims(::Type{D}) where {D<:AbstractDeformation} = ndims(supertype(D))
Base.eltype(d::AbstractDeformation) = eltype(typeof(d))
Base.ndims(d::AbstractDeformation) = ndims(typeof(d))

"""
`ϕ = GridDeformation(u::Array{<:SVector}, dims)` creates a
deformation `ϕ` for an array of size `dims`.  `u` specifies the
"pixel-wise" displacement at a series of control points that are
evenly-spaced over the domain specified by `dims` (i.e., using
knot-vectors `linspace(1,dims[d],size(u,d))`).  In particular, each
corner of the array is the site of one control point.

`ϕ = GridDeformation(u::Array{<:SVector}, knots)` specifies the
knot-vectors manually. `u` must have dimensions equal to
`(length(knots[1]), length(knots[2]), ...)`.

`ϕ = GridDeformation(u::Array{T<:Real}, ...)` constructs the
deformation from a "plain" array. For a deformation in `N` dimensions,
`u` must have `N+1` dimensions, where the first dimension corresponds
to the displacement along each axis (and therefore `size(u,1) == N`).

Finally, `ϕ = GridDeformation((u1, u2, ...), ...)` allows you to
construct the deformation using an `N`-tuple of shift-arrays, each
with `N` dimensions.
"""
struct GridDeformation{T,N,A<:AbstractArray,L} <: AbstractDeformation{T,N}
    u::A
    knots::NTuple{N,L}

    function GridDeformation{T,N,A,L}(u::AbstractArray{FV,N}, knots::NTuple{N,L}) where {T,N,A,L,FV<:SVector}
        typeof(u) == A || error("typeof(u) = $(typeof(u)), which is different from $A")
        length(FV) == N || throw(DimensionMismatch("Dimensionality $(length(FV)) must match $N knot vectors"))
        for d = 1:N
            size(u, d) == length(knots[d]) || error("size(u) = $(size(u)), but the knots specify a grid of size $(map(length, knots))")
        end
        new{T,N,A,L}(u, knots)
    end
    function GridDeformation{T,N,A,L}(u::ScaledInterpolation{FV,N}) where {T,N,A,L,FV<:SVector}
        new{T,N,A,L}(u, u.ranges)
    end
end

const InterpolatingDeformation{T,N,A<:AbstractInterpolation} = GridDeformation{T,N,A}

# Ambiguity avoidance
function GridDeformation(u::AbstractArray{FV,N},
                         knots::Tuple{}) where {FV<:SVector,N}
    error("Cannot supply an empty knot tuple")
end

# With knot ranges
function GridDeformation(u::AbstractArray{FV,N},
                             knots::NTuple{N,L}) where {FV<:SVector,N,L<:AbstractVector}
    T = eltype(FV)
    length(FV) == N || throw(DimensionMismatch("$N-dimensional array requires SVector{$N,T}"))
    GridDeformation{T,N,typeof(u),L}(u, knots)
end

# With image spatial size
function GridDeformation(u::AbstractArray{FV,N},
                             dims::NTuple{N,L}) where {FV<:SVector,N,L<:Integer}
    T = eltype(FV)
    length(FV) == N || throw(DimensionMismatch("$N-dimensional array requires SVector{$N,T}"))
    knots = ntuple(d->range(1, stop=dims[d], length=size(u,d)), N)
    GridDeformation{T,N,typeof(u),typeof(knots[1])}(u, knots)
end

# Construct from a plain array
function GridDeformation(u::AbstractArray{T}, knots::NTuple{N}) where {T<:Number,N}
    ndims(u) == N+1 || error("Need $(N+1) dimensions for $N-dimensional deformations")
    size(u,1) == N || error("First dimension of u must be of length $N")
    uf = convert_to_fixed(SVector{N,T}, u, tail(size(u)))
    GridDeformation(uf, knots)
end

# Construct from a (u1, u2, ...) tuple
function GridDeformation(u::NTuple{N,AbstractArray}, knots::NTuple{N}) where N
    ndims(u[1]) == N || error("Need $N dimensions for $N-dimensional deformations")
    ua = permutedims(cat(N+1, u...), (N+1,(1:N)...))
    uf = convert_to_fixed(ua)
    GridDeformation(uf, knots)
end

# When knots is a vector
GridDeformation(u, knots::AbstractVector{V}) where {V<:AbstractVector} = GridDeformation(u, (knots...,))
GridDeformation(u, knots::AbstractVector{R}) where {R<:Real} = GridDeformation(u, (knots,))

function GridDeformation(u::ScaledInterpolation{FV}) where FV<:SVector
    N = length(FV)
    ndims(u) == N || throw(DimensionMismatch("Dimension $(ndims(u)) incompatible with vectors of length $N"))
    GridDeformation{eltype(FV),N,typeof(u),typeof(u.ranges[1])}(u)
end

"""
`ϕs = griddeformations(u, knots)` constructs a vector `ϕs` of
seqeuential deformations.  The last dimension of the array `u` should
correspond to time; `ϕs[i]` is produced from `u[..., i]`.
"""
function griddeformations(u::AbstractArray{T}, knots::NTuple{N}) where {N,T<:Number}
    ndims(u) == N+2 || error("Need $(N+2) dimensions for a vector of $N-dimensional deformations")
    size(u,1) == N || error("First dimension of u must be of length $N")
    uf = RegisterDeformation.convert_to_fixed(SVector{N,T}, u, Base.tail(size(u)))
    griddeformations(uf, knots)
end

function griddeformations(u::AbstractArray{FV}, knots::NTuple{N}) where {N,FV<:SVector}
    ndims(u) == N+1 || error("Need $(N+1) dimensions for a vector of $N-dimensional deformations")
    length(FV) == N || throw(DimensionMismatch("Dimensionality $(length(FV)) must match $N knot vectors"))
    colons = ntuple(d->Colon(), N)::NTuple{N,Colon}
    [GridDeformation(view(u, colons..., i), knots) for i = 1:size(u, N+1)]
end

Base.:(==)(ϕ1::GridDeformation, ϕ2::GridDeformation) = ϕ1.u == ϕ2.u && ϕ1.knots == ϕ2.knots

Base.copy(ϕ::GridDeformation{T,N,A,L}) where {T,N,A,L} = (u = copy(ϕ.u); GridDeformation{T,N,typeof(u),L}(u, map(copy, ϕ.knots)))

# # TODO: flesh this out
# immutable VoroiDeformation{T,N,Vu<:AbstractVector,Vc<:AbstractVector} <: AbstractDeformation{T,N}
#     u::Vu
#     centers::Vc
#     simplexes::??
# end
# (but there are several challenges, including the lack of a continuous gradient)

function Interpolations.interpolate(ϕ::GridDeformation, BC)
    itp = scale(interpolate(ϕ.u, BSpline(Quadratic(Flat(OnCell())))), ϕ.knots...)
    GridDeformation(itp)
end
Interpolations.interpolate(ϕ::GridDeformation) = interpolate(ϕ, Flat())

function Interpolations.interpolate!(ϕ::GridDeformation, BC)
    itp = scale(interpolate!(ϕ.u, BSpline(Quadratic(BC)), OnCell()), ϕ.knots...)
    GridDeformation(itp)
end
Interpolations.interpolate!(ϕ::GridDeformation) = interpolate!(ϕ, InPlace())

Interpolations.interpolate(ϕ::GridDeformation{T,N,A}) where { T,N,A<:AbstractInterpolation} = error("ϕ is already interpolating")

Interpolations.interpolate!(ϕ::GridDeformation{T,N,A}) where {T,N,A<:AbstractInterpolation} = error("ϕ is already interpolating")

function vecindex(ϕ::GridDeformation{T,N,A}, x::SVector{N}) where {T,N,A<:AbstractInterpolation}
    x + vecindex(ϕ.u, x)
end

@generated function Base.getindex(ϕ::GridDeformation{T,N,A}, xs::Number...) where {T,N,A<:AbstractInterpolation}
    length(xs) == N || throw(DimensionMismatch("$(length(xs)) indexes is not consistent with ϕ dimensionality $N"))
    xindexes = [:(xs[$d]) for d = 1:N]
    ϕxindexes = [:(xs[$d]+ux[$d]) for d = 1:N]
    meta = Expr(:meta, :inline)
    quote
        $meta
        ux = ϕ.u[$(xindexes...)]
        SVector($(ϕxindexes...))
    end
end

# Composition ϕ_old(ϕ_new(x))
function (ϕ_old::GridDeformation{T1,N,A})(ϕ_new::GridDeformation{T2,N}) where {T1,T2,N,A<:AbstractInterpolation}
    uold, knots = ϕ_old.u, ϕ_old.knots
    if !isa(ϕ_new.u, AbstractInterpolation)
        ϕ_new.knots == knots || error("If knots are incommensurate, ϕ_new must be interpolating")
    end
    ucomp = _compose(uold, ϕ_new.u, knots)
    GridDeformation(ucomp, knots)
end

(ϕ_old::GridDeformation)(ϕ_new::GridDeformation) =
    error("ϕ_old must be interpolating")

function _compose(uold, unew, knots)
    sz = map(length, knots)
    x = knot(knots, 1)
    out = _compose(uold, unew, x, 1)
    ucomp = similar(uold, typeof(out))
    for I in CartesianIndices(sz)
        ucomp[I] = _compose(uold, unew, knot(knots, I), I)
    end
    ucomp
end

function _compose(uold, unew, x, i)
    dx = lookup(unew, x, i)
    dx + vecindex(uold, x+dx)
end

lookup(u::AbstractInterpolation, x, i) = vecindex(u, x)
lookup(u, x, i) = u[i]

@generated function knot(knots::NTuple{N}, i::Integer) where N
    args = [:(knots[$d][s[$d]]) for d = 1:N]
    quote
        s = CartesianIndices(map(length, knots))[i] # ind2sub(map(length, knots), i)
        SVector($(args...))
    end
end

@generated function knot(knots::NTuple{N}, I) where N
    args = [:(knots[$d][I[$d]]) for d = 1:N]
    :(SVector($(args...)))
end

arraysize(knots::NTuple) = map(k->(x = extrema(k); convert(Int, x[2]-x[1]+1)), knots)

struct KnotIterator{K,N}
    knots::K
    iter::CartesianIndices{N}
end

eachknot(knots) = KnotIterator(knots, CartesianIndices(map(length, knots)))

Base.start(ki::KnotIterator) = start(ki.iter)
Base.done(ki::KnotIterator, state) = done(ki.iter, state)
function Base.next(ki::KnotIterator, state)
    I, state = next(ki.iter, state)
    k = knot(ki.knots, I)
    k, state
end

"""
    ϕnew = regrid(ϕ, gridsize)

Reparametrize the deformation `ϕ` so that it has a grid size `gridsize`.

# Example

```
ϕnew = regrid(ϕ, (13,11,7))
```
for a 3-dimensional deformation `ϕ`.
"""
function regrid(ϕ::InterpolatingDeformation{T,N}, sz::Dims{N}) where {T,N}
    knots_new = map((r,n)->range(first(r), stop=last(r), length=n), ϕ.knots, sz)
    u = Array{SVector{N,T},N}(undef, sz)
    for (i, k) in enumerate(eachknot(knots_new))
        u[i] = ϕ.u[k...]
    end
    GridDeformation(u, knots_new)
end
regrid(ϕ::GridDeformation{T,N}, sz::Dims{N}) where {T,N} = regrid(interpolate(ϕ), sz)

"""
`ϕ_c = ϕ_old(ϕ_new)` computes the composition of two deformations,
yielding a deformation for which `ϕ_c(x) ≈ ϕ_old(ϕ_new(x))`. `ϕ_old`
must be interpolating (see `interpolate(ϕ_old)`).

`ϕ_c, g = compose(ϕ_old, ϕ_new)` also yields the gradient `g` of `ϕ_c`
with respect to `u_new`.  `g[i,j,...]` is the Jacobian matrix at grid
position `(i,j,...)`.

You can use `_, g = compose(identity, ϕ_new)` if you need the gradient
for when `ϕ_old` is equal to the identity transformation.
"""
function compose(ϕ_old::GridDeformation{T1,N,A}, ϕ_new::GridDeformation{T2,N}) where {T1,T2,N,A<:AbstractInterpolation}
    u, knots = ϕ_old.u, ϕ_old.knots
    ϕ_new.knots == knots || error("Not yet implemented for incommensurate knots")
    unew = ϕ_new.u
    sz = map(length, knots)
    x = knot(knots, 1)
    out = _compose(u, unew, x, 1)
    ucomp = similar(u, typeof(out))
    TG = similar_type(SArray, eltype(out), Size(N, N))
    g = Array{TG}(undef, size(u))
    gtmp = Vector{typeof(out)}(undef, N)
    eyeN = TG(1.0I) # eye(TG)
    for I in CartesianIndices(sz)
        x = knot(knots, I)
        dx = lookup(unew, x, I)
        y = x + dx
        ucomp[I] = dx + vecindex(u, y)
        vecgradient!(gtmp, u, y)
        g[I] = hcat(ntuple(d->gtmp[d], Val(N))...) + eyeN
    end
    GridDeformation(ucomp, knots), g
end

"""
`ϕsi_old` and `ϕs_new` will generated `ϕs_c` vector and `g` vector
`ϕsi_old` is interpolated ϕs_old:
e.g) `ϕsi_old = map(Interpolations.interpolate!, copy(ϕs_old))`
"""
function compose(ϕsi_old::AbstractVector{G1}, ϕs_new::AbstractVector{G2}) where {G1<:GridDeformation, G2<:GridDeformation}
    n = length(ϕs_new)
    length(ϕsi_old) == n || throw(DimensionMismatch("vectors-of-deformations must have the same length, got $(length(ϕsi_old)) and $n"))
    ϕc1, g1 = compose(first(ϕsi_old), first(ϕs_new))
    ϕs_c = Vector{typeof(ϕc1)}(undef, n)
    gs = Vector{typeof(g1)}(undef, n)
    ϕs_c[1], gs[1] = ϕc1, g1
    for i in 2:n
        ϕs_c[i], gs[i] = compose(ϕsi_old[i], ϕs_new[i]);
    end
    ϕs_c, gs
end


function compose(f::Function, ϕ_new::GridDeformation{T,N}) where {T,N}
    f == identity || error("Only the identity function is supported")
    ϕ_new, fill(similar_type(SArray, T, Size(N, N))(1.0I), size(ϕ_new.u))
end

function medfilt(ϕs::AbstractVector{D}, n) where D<:AbstractDeformation
    nhalf = n>>1
    2nhalf+1 == n || error("filter size must be odd")
    T = eltype(eltype(D))
    v = Array{T}(undef, ndims(D), n)
    vs = ntuple(d->view(v, d, :), ndims(D))
    ϕ1 = copy(ϕs[1])
    ϕout = Vector{typeof(ϕ1)}(undef, length(ϕs))
    ϕout[1] = ϕ1
    _medfilt!(ϕout, ϕs, v, vs)  # function barrier due to instability of vs
end

@noinline function _medfilt!(ϕout, ϕs, v, vs::NTuple{N,T}) where {N,T}
    n = size(v,2)
    nhalf = n>>1
    tmp = Vector{eltype(T)}(undef, N)
    u1 = ϕout[1].u
    for i = 1+nhalf:length(ϕs)-nhalf
        u = similar(u1)
        for I in eachindex(ϕs[i].u)
            for j = -nhalf:nhalf
                utmp = ϕs[i+j].u[I]
                for d = 1:N
                    v[d, j+nhalf+1] = utmp[d]
                end
            end
            for d = 1:N
                tmp[d] = median!(vs[d])
            end
            u[I] = tmp
        end
        ϕout[i] = GridDeformation(u, ϕs[i].knots)
    end
    # Copy the beginning and end
    for i = 2:nhalf  # we did [1] in medfilt
        ϕout[i] = copy(ϕs[i])
    end
    for i = length(ϕs)-nhalf+1:length(ϕs)
        ϕout[i] = copy(ϕs[i])
    end
    ϕout
end

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
mutable struct WarpedArray{T,N,A<:Extrapolatable,D<:AbstractDeformation} <: AbstractArray{T,N}
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
    itp = scale(interpolate(ϕ.u, BSpline(Quadratic(Flat())), OnCell()), ϕ.knots...)
    ϕ′ = GridDeformation(itp, ϕ.knots)
    WarpedArray{T,N,typeof(data),typeof(ϕ′)}(data, ϕ′)
end

WarpedArray(data, ϕ::GridDeformation) = WarpedArray(to_etp(data), ϕ)


Base.size(A::WarpedArray) = size(A.data)
Base.size(A::WarpedArray, i::Integer) = size(A.data, i)

@generated function Base.getindex(W::WarpedArray{T,N}, x::Number...) where {T,N}
    length(x) == N || error("Must use $N indexes")
    getindex_impl(N)
end

function getindex_impl(N)
    indxx = [:(x[$d]) for d = 1:N]
    indxϕx = [:(ϕx[$d]) for d = 1:N]
    meta = Expr(:meta, :inline)
    quote
        $meta
        ϕx = W.ϕ[$(indxx...)]
        W.data[$(indxϕx...)]
    end
end

getindex!(dest, W::WarpedArray, coords...) = Base._unsafe_getindex!(dest, W, coords...)

"""
`Atrans = translate(A, displacement)` shifts `A` by an amount
specified by `displacement`.  Specifically, in simple cases `Atrans[i,
j, ...] = A[i+displacement[1], j+displacement[2], ...]`.  More
generally, `displacement` is applied only to the spatial coordinates
of `A`; if `A` is an `Image`, dimensions marked as time or color are
unaffected.

`NaN` is filled in for any missing pixels.
"""
function translate(A::AbstractArray, displacement::DimsLike)
    disp = zeros(Int, ndims(A))
    disp[[coords_spatial(A)...]] = displacement
    indx = UnitRange{Int}[ (1:size(A,i))+disp[i] for i = 1:ndims(A) ]
    get(A, indx, NaN)
end

"""
`ϕ = tform2deformation(tform, arraysize, gridsize)` constructs a deformation
`ϕ` from the affine transform `tform` suitable for warping arrays
of size `arraysize`.  The origin-of-coordinates for `tform` is the
center of the array, meaning that if `tform` is a pure rotation the
array "spins" around its center.  The array of grid points defining `ϕ` has
size specified by `gridsize`.  The dimensionality of `tform` must
match that specified by `arraysize` and `gridsize`.
"""
function tform2deformation(tform::AffineMap{M,V}, arraysize, gridsize) where {M,V}
    N = length(size(tform.linear))
    if length(arraysize) != N || length(gridsize) != N
        error("Dimensionality mismatch")
    end
    A = tform.linear - Matrix{Float64}(I,N,N)  # this will compute the difference
    ngrid = prod(gridsize)
    u = Matrix{eltype(M)}(undef, N, ngrid)
    asz = [arraysize...]
    s = (asz.-1)./([gridsize...].-1)
    k = 0
    center = (asz.-1)/2  # adjusted for unit-offset
    for c in Counter(gridsize)
        x = (c.-1).*s - center
        u[:,k+=1] = A*x+tform.translation
    end
    urs = reshape(u, N, gridsize...)
    knots = ntuple(d->range(1, stop=arraysize[d], length=gridsize[d]), N)
    GridDeformation(urs, knots)
end

"""
`wimg = warp(img, ϕ)` warps the array `img` according to the
deformation `ϕ`.
"""
function warp(img::AbstractArray, ϕ::AbstractDeformation)
    wimg = WarpedArray(img, ϕ)
    dest = similar(img, warp_type(img))
    warp!(dest, wimg)
end

warp_type(img::AbstractArray{T}) where {T<:AbstractFloat} = T
warp_type(img::AbstractArray{T}) where {T<:Number} = Float32
warp_type(img::AbstractArray{C}) where {C<:Colorant} = warp_type(img, eltype(eltype(C)))
warp_type(img::AbstractArray{C}, ::Type{T}) where {C<:Colorant, T<:AbstractFloat} = C
warp_type(img::AbstractArray{C}, ::Type{T}) where {C<:Colorant, T} = base_colorant_type(C){Float32}

"""
`warp!(dest, src::WarpedArray)` instantiates a `WarpedArray` in the output `dest`.
"""
@generated function warp!(dest::AbstractArray{T,N}, src::WarpedArray) where {T,N}
    ϕxindexes = [:(I[$d]+ux[$d]) for d = 1:N]
    quote
        size(dest) == size(src) || error("dest must have the same size as src")
        # Can use zip once MAX_TYPE_DEPTH gets bumped, see julia #13561
        destiter = CartesianRange(size(dest))
        deststate = start(destiter)
        for ux in eachvalue(src.ϕ.u)
            I, deststate = next(destiter, deststate)
            dest[I] = src.data[$(ϕxindexes...)]
        end
        dest
    end
end


"""
`warp!(dest, img, ϕ)` warps `img` using the deformation `ϕ`.  The
result is stored in `dest`.
"""
function warp!(dest::AbstractArray, img::AbstractArray, ϕ::AbstractDeformation)
    wimg = WarpedArray(to_etp(img), ϕ)
    warp!(dest, wimg)
end

"""
`warp!(dest, img, tform, ϕ)` warps `img` using a combination of the affine transformation `tform` followed by deformation with `ϕ`.  The result is stored in `dest`.
"""
function warp!(dest::AbstractArray, img::AbstractArray, A::AffineMap, ϕ::AbstractDeformation)
    wimg = WarpedArray(to_etp(img, A), ϕ)
    warp!(dest, wimg)
end

"""

`warp!(T, io, img, ϕs; [nworkers=1])` writes warped images to
disk. `io` is an `IO` object or HDF5/JLD2 dataset (the latter must be
pre-allocated using `d_create` to be of the proper size). `img` is an
image sequence, and `ϕs` is a vector of deformations, one per image in
`img`.  If `nworkers` is greater than one, it will spawn additional
processes to perform the deformation.

An alternative syntax is `warp!(T, io, img, uarray; [nworkers=1])`,
where `uarray` is an array of `u` values with `size(uarray)[end] ==
nimages(img)`.
"""
function warp!(::Type{T}, dest::Union{IO,HDF5Dataset,JLD2.JLDFile}, img, ϕs; nworkers=1) where T
    n = nimages(img)
    ssz = size(img, coords_spatial(img)...)
    if n == 1
        ϕ = extract1(ϕs, sdims(img), ssz)
        destarray = Array{T}(undef, ssz)
        warp!(destarray, img, ϕ)
        warp_write(dest, destarray)
        return nothing
    end
    checkϕdims(ϕs, sdims(img), n)
    if nworkers > 1
        return _warp!(T, dest, img, ϕs, nworkers)
    end
    destarray = Array{T}(undef, ssz)
    @showprogress 1 "Stacks:" for i = 1:n
        ϕ = extracti(ϕs, i, ssz)
        warp!(destarray, view(img, timeaxis(img)(i)), ϕ)
        warp_write(dest, destarray, i)
    end
    nothing
end

warp!(::Type{T}, dest::Union{IO,HDF5Dataset,JLD2.JLDFile}, img, u::AbstractArray{R}; kwargs...) where {T,R<:Real} = warp!(T, dest, img, convert_to_fixed(u); kwargs...)

warp!(dest::Union{HDF5Dataset,JLD2.JLDFile}, img, u; nworkers=1) =
    warp!(eltype(dest), dest, img, u; nworkers=nworkers)

function _warp!(::Type{T}, dest, img, ϕs, nworkers) where T
    n = nimages(img)
    ssz = size(img, coords_spatial(img)...)
    wpids = addprocs(nworkers)
    simg = Vector{Any}()
    swarped = Vector{Any}()
    rrs = Vector{RemoteChannel}()
    mydir = splitdir(@__FILE__)[1]
    for p in wpids
        remotecall_fetch(Main.eval, p, :(push!(LOAD_PATH, $mydir)))
        remotecall_fetch(Main.eval, p, :(using RegisterDeformation))
        push!(simg, SharedArray{eltype(img)}(ssz, pids=[myid(),p]))
        push!(swarped, SharedArray{T}(ssz, pids=[myid(),p]))
    end
    nextidx = 0
    getnextidx() = nextidx += 1
    writing_mutex = RemoteChannel()
    prog = Progress(n, 1, "Stacks:")
    @sync begin
        for i = 1:nworkers
            p = wpids[i]
            src = simg[i]
            warped = swarped[i]
            @async begin
                while (idx = getnextidx()) <= n
                    ϕ = extracti(ϕs, idx, ssz)
                    copy!(src, view(img, timeaxis(img)(idx)))
                    remotecall_fetch(warp!, p, warped, src, ϕ)
                    put!(writing_mutex, true)
                    warp_write(dest, warped, idx)
                    update!(prog, idx)
                    take!(writing_mutex)
                end
            end
        end
    end
    finish!(prog)
    nothing
end

warp_write(io::IO, destarray) = write(io, destarray)
function warp_write(io::IO, destarray, i)
    offset = (i-1)*length(destarray)*sizeof(eltype(destarray))
    seek(io, offset)
    write(io, destarray)
end
function warp_write(dest, destarray, i)
    colons = [Colon() for d = 1:ndims(destarray)]
    dest[colons..., i] = destarray
end

function extract1(u::AbstractArray{V}, N, ssz) where V<:SVector
    if ndims(u) == N+1
        ϕ = GridDeformation(reshape(u, size(u)[1:end-1]), ssz)
    else
        ϕ = GridDeformation(u, ssz)
    end
    ϕ
end
extract1(ϕs::Vector{D}, N, ssz) where {D<:AbstractDeformation} = ϕs[1]

function extracti(u::AbstractArray{V}, i, ssz) where V<:SVector
    colons = [Colon() for d = 1:ndims(u)-1]
    GridDeformation(u[colons..., i], ssz)
end
extracti(ϕs::Vector{D}, i, _) where {D<:AbstractDeformation} = ϕs[i]

function checkϕdims(u::AbstractArray{V}, N, n) where V<:SVector
    ndims(u) == N+1 || error("u's dimensionality $(ndims(u)) is inconsistent with the number of spatial dimensions $N of the image")
    if size(u)[end] != n
        error("Must have one `u` slice per image")
    end
    nothing
end
checkϕdims(ϕs::Vector{D}, N, n) where {D<:AbstractDeformation} = length(ϕs) == n || error("Must have one `ϕ` per image")


"""
    img = knotgrid(T, knots)
    img = knotgrid(knots)
    img = knotgrid([T], ϕ)

Returns an image `img` which has value 1 at points that are "on the
knots."  If `knots`/`ϕ` is two-dimensional, the image will
consist of grid lines; for three-dimensional inputs, the image will
have grid planes.

The meaning of "on the knots" is that `x[d] == knots[d][i]` for some
dimension `d` and knot index `i`. An exception is made for the edge
values, which are brought inward by one pixel to prevent complete loss
under subpixel deformations.

Optionally specify the element type `T` of `img`.

See also [`warpgrid`](@ref).
"""
function knotgrid(::Type{T}, knots::NTuple{N,AbstractRange}) where {T,N}
    @assert all(r->first(r)==1, knots)
    inds = map(r->Base.OneTo(ceil(Int, last(r))), knots)
    img = Array{T}(undef, map(length, inds))
    fill!(img, zero(T))
    for idim = 1:N
        indexes = Any[inds...]
        indexes[idim] = map(x->clamp(round(Int, x), first(inds[idim])+1, last(inds[idim])-1), knots[idim])
        img[indexes...] = one(T)
    end
    img
end
knotgrid(::Type{T}, ϕ::GridDeformation) where {T} = knotgrid(T, ϕ.knots)
knotgrid(arg) = knotgrid(Bool, arg)


"""
    img = warpgrid(ϕ; [scale=1, showidentity=false])

Returns an image `img` that permits visualization of the deformation
`ϕ`.  The output is a warped rectangular grid with nodes centered on
the control points as specified by the knots of `ϕ`.  If `ϕ` is a
two-dimensional deformation, the image will consist of grid lines; for
a three-dimensional deformation, the image will have grid planes.

`scale` multiplies `ϕ.u`, effectively making the deformation stronger
(for `scale > 1`).  This can be useful if you are trying to visualize
subtle changes. If `showidentity` is `true`, an RGB image is returned
that has the warped grid in magenta and the original grid in green.

See also [`knotgrid`](@ref).
"""
function warpgrid(ϕ; scale=1, showidentity::Bool=false)
    img = knotgrid(ϕ)
    if scale != 1
        ϕ = GridDeformation(scale*ϕ.u, ϕ.knots)
    end
    wimg = warp(img, ϕ)
    if showidentity
        n = ndims(img)+1
        return reinterpret(RGB{Float32}, permutedims(cat(n, wimg, img, wimg), (n,1:ndims(img)...)))
    end
    wimg
end

"""
`ϕs = tinterpolate(ϕsindex, tindex, nstack)` uses linear
interpolation/extrapolation in time to "fill out" to times `1:nstack`
a deformation defined intermediate times `tindex` . Note that
`ϕs[tindex] == ϕsindex`.
"""
function tinterpolate(ϕsindex, tindex, nstack)
    ϕs = Vector{eltype(ϕsindex)}(undef, nstack)
    # Before the first tindex
    k = 0
    for i in 1:tindex[1]-1
        ϕs[k+=1] = ϕsindex[1]
    end
    # Within tindex
    for i in 2:length(tindex)
        ϕ1 = ϕsindex[i-1]
        ϕ2 = ϕsindex[i]
        Δt = tindex[i] - tindex[i-1]
        for j = 0:Δt-1
            α = convert(eltype(eltype(ϕ1.u)), j/Δt)
            ϕs[k+=1] = GridDeformation((1-α)*ϕ1.u + α*ϕ2.u, ϕ2.knots)
        end
    end
    # After the last tindex
    for i in tindex[end]:nstack
        ϕs[k+=1] = ϕsindex[end]
    end
    return ϕs
end

# TODO?: do we need to return real values beyond-the-edge for a SubArray?
to_etp(img) = extrapolate(interpolate(img, BSpline(Linear()), OnGrid()), convert(promote_type(eltype(img), Float32), NaN))

to_etp(itp::AbstractInterpolation) = extrapolate(itp, convert(promote_type(eltype(itp), Float32), NaN))

to_etp(etp::AbstractExtrapolation) = etp

to_etp(img, A::AffineMap) = TransformedArray(to_etp(img), A)

# Extensions to Interpolations and StaticArrays
@generated function vecindex(A::AbstractArray, x::SVector{N}) where N
    args = [:(x[$d]) for d = 1:N]
    meta = Expr(:meta, :inline)
    quote
        $meta
        getindex(A, $(args...))
    end
end

@generated function vecindex(A::AbstractInterpolation, x::SVector{N}) where N
    args = [:(x[$d]) for d = 1:N]
    meta = Expr(:meta, :inline)
    quote
        $meta
        A($(args...))
    end
end

@generated function vecgradient!(g, itp::AbstractArray, x::SVector{N}) where N
    args = [:(x[$d]) for d = 1:N]
    meta = Expr(:meta, :inline)
    quote
        $meta
        Interpolations.gradient!(g, itp, $(args...))
    end
end

function convert_to_fixed(u::Array{T}, sz=size(u)) where T
    N = sz[1]
    convert_to_fixed(SVector{N, T}, u, tail(sz))
end

# Unlike the one above, this is type-stable
function convert_to_fixed(::Type{SVector{N,T}}, u::AbstractArray{T}, sz=tail(size(u))) where {T,N}
    if isbitstype(T)
        uf = reshape(reinterpret(SVector{N,T}, vec(u)), sz)
    else
        uf = Array{SVector{N,T}}(undef, sz)
        copy_ctf!(uf, u)
    end
    uf
end

@generated function copy_ctf!(dest::Array{SVector{N,T}}, src::Array) where {N,T}
    exvec = [:(src[offset+$d]) for d=1:N]
    quote
        for i = 1:length(dest)
            offset = (i-1)*N
            dest[i] = SVector($(exvec...))
        end
        dest
    end
end

function convert_from_fixed(uf::AbstractArray{SVector{N,T}}, sz=size(uf)) where {N,T}
    if isbitstype(T) && isa(uf, Array)
        u = reshape(reinterpret(T, vec(uf)), (N, sz...))
    else
        u = Array{T}(undef, N, sz...)
        for i = 1:length(uf)
            for d = 1:N
                u[d,i] = uf[i][d]
            end
        end
    end
    u
end

# # Note this is a bit unsafe as it requires the user to specify C correctly
# @generated function Base.convert{R,C,T}(::Type{SMatrix{Tuple{R,C},T}}, v::Vector{SVector{R,T}})
#     args = [:(Tuple(v[$d])) for d = 1:C]
#     :(SMatrix{Tuple{R,C},T}(($(args...),)))
# end

include_string(RegisterDeformation, """
    const RowVectorArray{T} = RowVector{T,Vector{T}}

    Base.reinterpret(::Type{T}, r::RowVectorArray{S}) where {T,S} = reinterpret(T, r.vec)
    Base.reinterpret(::Type{T}, r::RowVectorArray{S}, dims::Dims{N}) where {T,S,N} = reshape(reinterpret(T, r.vec), dims)
    """)

# Wrapping functions to interface with CoordinateTransfromations instead of AffineTransfroms module
tformtranslate(trans::Vector) = (m = length(trans); AffineMap(Matrix{Float64}(I,m,m), trans))

rotation2(angle) = [cos(angle) -sin(angle); sin(angle) cos(angle)]
function tformrotate(angle)
    A = rotation2(angle)
    AffineMap(A, zeros(eltype(A),2))
end

#=
function tformrotate(axis::Vector, angle)
    if length(axis) == 3
        return AffineTransform(rotation3(axis, angle), zeros(eltype(axis),3))
    else
        error("Dimensionality ", length(axis), " not supported")
    end
end

function tformrotate(x::Vector)
    if length(x) == 3
        return AffineTransform(rotation3(x), zeros(eltype(x),3))
    else
        error("Dimensionality ", length(x), " not supported")
    end
end


# The following assumes uaxis is normalized
function _rotation3(uaxis::Vector, angle)
    if length(uaxis) != 3
        error("3d rotations only")
    end
    ux, uy, uz = uaxis[1], uaxis[2], uaxis[3]
    c = cos(angle)
    s = sin(angle)
    cm = one(typeof(c)) - c
    R = [c+ux*ux*cm       ux*uy*cm-uz*s      ux*uz*cm+uy*s;
         uy*ux*cm+uz*s      c+uy*uy*cm       uy*uz*cm-ux*s;
         uz*ux*cm-uy*s    uz*uy*cm+ux*s      c+uz*uz*cm]
end

function rotation3{T}(axis::Vector{T}, angle)
    n = norm(axis)
    axisn = n>0 ? axis/n : (tmp = zeros(T,length(axis)); tmp[1] = 1)
    _rotation3(axisn, angle)
end

# Angle/axis representation where the angle is the norm of the vector (so axis is not normalized)
function rotation3{T}(axis::Vector{T})
    n = norm(axis)
    axisn = n>0 ? axis/n : (tmp = zeros(typeof(one(T)/1),length(axis)); tmp[1] = 1; tmp)
    _rotation3(axisn, n)
end
=#
end  # module
