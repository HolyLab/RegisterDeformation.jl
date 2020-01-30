"""
`ϕ = GridDeformation(u::Array{<:SVector}, axs)` creates a
deformation `ϕ` for an array with axes `axs`.  `u` specifies the
"pixel-wise" displacement at a series of nodes that are
evenly-spaced over the domain specified by `axs` (i.e., using
node-vectors `range(first(axs[d]), stop=last(axs[d]), length=size(u,d))`).
In particular, each corner of the array is the site of one node.

`ϕ = GridDeformation(u::Array{<:SVector}, nodes)` specifies the
node-vectors manually. `u` must have dimensions equal to
`(length(nodes[1]), length(nodes[2]), ...)`.

`ϕ = GridDeformation(u::Array{T<:Real}, ...)` constructs the
deformation from a "plain" `u` array. For a deformation in `N` dimensions,
`u` must have `N+1` dimensions, where the first dimension corresponds
to the displacement along each axis (and therefore `size(u,1) == N`).

Finally, `ϕ = GridDeformation((u1, u2, ...), ...)` allows you to
construct the deformation using an `N`-tuple of shift-arrays, each
with `N` dimensions.

# Example

To represent a two-dimensional deformation over a spatial region `1:100 × 1:200`
(e.g., for an image of that size),

```julia
gridsize = (3, 5)             # a coarse grid
u = 10*randn(2, gridsize...)  # each displacement is 2-dimensional, typically ~10 pixels
nodes = (range(1, stop=100, length=gridsize[1]), range(1, stop=200, length=gridsize[2]))
ϕ = GridDeformation(u, nodes) # this is a "naive" deformation (not ready for interpolation)
```
"""
struct GridDeformation{T,N,A<:AbstractArray,L} <: AbstractDeformation{T,N}
    u::A
    nodes::NTuple{N,L}

    function GridDeformation{T,N,A,L}(u::AbstractArray{FV,N}, nodes::NTuple{N,L}) where {T,N,A,L,FV<:SVector}
        typeof(u) == A || error("typeof(u) = $(typeof(u)), which is different from $A")
        length(FV) == N || throw(DimensionMismatch("Dimensionality $(length(FV)) must match $N node vectors"))
        for d = 1:N
            size(u, d) == length(nodes[d]) || error("size(u) = $(size(u)), but the nodes specify a grid of size $(map(length, nodes))")
        end
        new{T,N,A,L}(u, nodes)
    end
    function GridDeformation{T,N,A,L}(u::ScaledInterpolation{FV,N}) where {T,N,A,L,FV<:SVector}
        new{T,N,A,L}(u, u.ranges)
    end
end

const InterpolatingDeformation{T,N,A<:AbstractInterpolation} = GridDeformation{T,N,A}

# With node ranges
function GridDeformation(u::AbstractArray{FV,N},
                         nodes::NTuple{N,L}) where {FV<:SVector,N,L<:AbstractVector}
    T = eltype(FV)
    length(FV) == N || throw(DimensionMismatch("$N-dimensional array requires SVector{$N,T}"))
    GridDeformation{T,N,typeof(u),L}(u, nodes)
end

# With image axes
function GridDeformation(u::AbstractArray{FV,N},
                         axs::NTuple{N,L}) where {FV<:SVector,N,L<:AbstractUnitRange{<:Integer}}
    T = eltype(FV)
    length(FV) == N || throw(DimensionMismatch("$N-dimensional array requires SVector{$N,T}"))
    nodes = ntuple(N) do d
        ax = axs[d]
        range(first(ax), stop=last(ax), length=size(u,d))
    end
    GridDeformation{T,N,typeof(u),typeof(nodes[1])}(u, nodes)
end

# Construct from a plain array
function GridDeformation(u::AbstractArray{T}, nodes::NTuple{N}) where {T<:Number,N}
    ndims(u) == N+1 || error("`u` needs $(N+1) dimensions for $N-dimensional deformations")
    size(u, 1) == N || error("first dimension of u must be of length $N")
    uf = Array(convert_to_fixed(SVector{N,T}, u, tail(size(u))))
    GridDeformation(uf, nodes)
end

# Construct from a (u1, u2, ...) tuple
function GridDeformation(u::NTuple{N,AbstractArray}, nodes::NTuple{N}) where N
    ndims(u[1]) == N || error("Need $N dimensions for $N-dimensional deformations")
    ua = permutedims(cat(u..., dims=N+1), (N+1,(1:N)...))
    uf = Array(convert_to_fixed(ua))
    GridDeformation(uf, nodes)
end

# When nodes is a vector
GridDeformation(u, nodes::AbstractVector{V}) where {V<:AbstractVector} = GridDeformation(u, (nodes...,))

function GridDeformation(u::ScaledInterpolation{FV}) where FV<:SVector
    N = length(FV)
    ndims(u) == N || throw(DimensionMismatch("Dimension $(ndims(u)) incompatible with vectors of length $N"))
    GridDeformation{eltype(FV),N,typeof(u),typeof(u.ranges[1])}(u)
end

function Base.show(io::IO, ϕ::GridDeformation{T}) where T
    if ϕ.u isa AbstractInterpolation
        print(io, "Interpolating ")
    end
    print(io, Base.dims2string(size(ϕ.u)), " GridDeformation{", T, "} over a domain ")
    for (i, n) in enumerate(ϕ.nodes)
        print(io, first(n), "..", last(n))
        i < length(ϕ.nodes) && print(io, '×')
    end
end

"""
`ϕs = griddeformations(u, nodes)` constructs a vector `ϕs` of
seqeuential deformations.  The last dimension of the array `u` should
correspond to time; `ϕs[i]` is produced from `u[:, ..., i]`.
"""
function griddeformations(u::AbstractArray{T}, nodes::NTuple{N}) where {N,T<:Number}
    ndims(u) == N+2 || error("Need $(N+2) dimensions for a vector of $N-dimensional deformations")
    size(u,1) == N || error("First dimension of u must be of length $N")
    uf = Array(convert_to_fixed(SVector{N,T}, u, Base.tail(size(u))))
    griddeformations(uf, nodes)
end

function griddeformations(u::AbstractArray{FV}, nodes::NTuple{N}) where {N,FV<:SVector}
    ndims(u) == N+1 || error("Need $(N+1) dimensions for a vector of $N-dimensional deformations")
    length(FV) == N || throw(DimensionMismatch("Dimensionality $(length(FV)) must match $N node vectors"))
    colons = ntuple(d->Colon(), Val(N))
    [GridDeformation(view(u, colons..., i), nodes) for i = 1:size(u, N+1)]
end

Base.:(==)(ϕ1::GridDeformation, ϕ2::GridDeformation) = ϕ1.u == ϕ2.u && ϕ1.nodes == ϕ2.nodes

Base.copy(ϕ::GridDeformation{T,N,A,L}) where {T,N,A,L} = (u = copy(ϕ.u); GridDeformation{T,N,typeof(u),L}(u, map(copy, ϕ.nodes)))

# # TODO: flesh this out
# immutable VoroiDeformation{T,N,Vu<:AbstractVector,Vc<:AbstractVector} <: AbstractDeformation{T,N}
#     u::Vu
#     centers::Vc
#     simplexes::??
# end
# (but there are several challenges, including the lack of a continuous gradient)


"""
    ϕi = interpolate(ϕ, BC=Flat(OnCell()))

Create a deformation `ϕi` suitable for interpolation, matching the displacements
of `ϕ.u` at the nodes. A quadratic interpolation scheme is used, with default
flat boundary conditions.
"""
function Interpolations.interpolate(ϕ::GridDeformation, BC=Flat(OnCell()))
    itp = scale(interpolate(ϕ.u, BSpline(Quadratic(BC))), ϕ.nodes...)
    GridDeformation(itp)
end

"""
    ϕi = interpolate!(ϕ, BC=InPlace(OnCell()))

Create a deformation `ϕi` suitable for interpolation, matching the displacements
of `ϕ.u` at the nodes. `ϕ` is destroyed in the process.

A quadratic interpolation scheme is used, with the default being to use "InPlace"
boundary conditions.

!!! warning
    Because of the difference in default boundary conditions,
    `interpolate!(copy(ϕ))` does *not* yield a result identical to `interpolate(ϕ)`.
    When it matters, it is recommended that you annotate such calls with
    `# not same as interpolate(ϕ)` in your code.
"""
function Interpolations.interpolate!(ϕ::GridDeformation, BC=InPlace(OnCell()))
    itp = scale(interpolate!(ϕ.u, BSpline(Quadratic(BC))), ϕ.nodes...)
    GridDeformation(itp)
end

Interpolations.interpolate(ϕ::InterpolatingDeformation, args...) = error("ϕ is already interpolating")
Interpolations.interpolate!(ϕ::InterpolatingDeformation, args...) = error("ϕ is already interpolating")

function vecindex(ϕ::GridDeformation{T,N,A}, x::SVector{N}) where {T,N,A<:AbstractInterpolation}
    x + vecindex(ϕ.u, x)
end

"""
    ϕ = similarϕ(ϕref, coefs)

Create a deformation with the same nodes as `ϕref` but using `coefs` for the data.
This is primarily useful for testing purposes, e.g., computing gradients with
`ForwardDiff` where the elements are `ForwardDiff.Dual` numbers.

If `ϕref` is interpolating, `coefs` will be used for the interpolation coefficients,
not the node displacements. Typically you want to create `ϕref` with

    ϕref = interpolate!(copy(ϕ0))

rather than `interpolate(ϕ0)` (see [`interpolate!`](@ref)).
"""
function similarϕ(ϕref, coefs::AbstractArray{<:Number})
    coefsref = getcoefs(ϕref)
    N = ndims(coefsref)
    udata = convert_to_fixed(SVector{N,eltype(coefs)}, coefs, size(coefsref))
    return similarϕ(ϕref, udata)
end

similarϕ(ϕref, udata::AbstractArray{<:SVector}) = _similarϕ(ϕref, udata)

_similarϕ(ϕref::GridDeformation, udata) = GridDeformation(udata, ϕref.nodes)

function _similarϕ(ϕref::InterpolatingDeformation, udata)
    scaleref = ϕref.u
    itpref = scaleref.itp
    itp = Interpolations.BSplineInterpolation(floattype(eltype(udata)), udata, itpref.it, itpref.parentaxes)
    return GridDeformation(scale(itp, ϕref.nodes...))
end

# @generated function Base.getindex(ϕ::GridDeformation{T,N,A}, xs::Vararg{Number,N}) where {T,N,A<:AbstractInterpolation}
#     xindexes = [:(xs[$d]) for d = 1:N]
#     ϕxindexes = [:(xs[$d]+ux[$d]) for d = 1:N]
#     quote
#         $(Expr(:meta, :inline))
#         ux = ϕ.u($(xindexes...))
#         SVector($(ϕxindexes...))
#     end
# end

@inline function (ϕ::GridDeformation{T,N,A})(xs::Vararg{Number,N}) where {T,N,A<:AbstractInterpolation}
    return ϕ.u(xs...) .+ xs
end

function (ϕ::GridDeformation{T,N})(xs::Vararg{Number,N}) where {T,N}
    error("call `ϕi = interpolate(ϕ)` and use `ϕi` for evaluating the deformation.")
end

@inline (ϕ::GridDeformation{T,N})(xs::SVector{N}) where {T,N} = ϕ(Tuple(xs)...)

# Composition ϕ_old(ϕ_new(x))
function (ϕ_old::GridDeformation{T1,N,A})(ϕ_new::GridDeformation{T2,N}) where {T1,T2,N,A<:AbstractInterpolation}
    uold, nodes = ϕ_old.u, ϕ_old.nodes
    if !isa(ϕ_new.u, AbstractInterpolation)
        ϕ_new.nodes == nodes || error("If nodes are incommensurate, ϕ_new must be interpolating")
    end
    ucomp = _compose(uold, ϕ_new.u, nodes)
    GridDeformation(ucomp, nodes)
end

(ϕ_old::GridDeformation)(ϕ_new::GridDeformation) =
    error("ϕ_old must be interpolating")

function _compose(uold, unew, nodes)
    sz = map(length, nodes)
    x = node(nodes, 1)
    out = _compose(uold, unew, x, 1)
    ucomp = similar(uold, typeof(out))
    for I in CartesianIndices(sz)
        ucomp[I] = _compose(uold, unew, node(nodes, I), I)
    end
    ucomp
end

function _compose(uold, unew, x, i)
    dx = lookup(unew, x, i)
    dx + vecindex(uold, x+dx)
end

lookup(u::AbstractInterpolation, x, i) = vecindex(extrapolate(u,Flat()), x) # using extrpolate to resolve BoundsError
lookup(u, x, i) = u[i]

@inline function node(nodes::NTuple{N}, i::Integer) where N
    I = CartesianIndices(map(length, nodes))[i]
    return node(nodes, I)
end

@inline function node(nodes::NTuple{N}, I::CartesianIndex{N}) where N
    return SVector(map(getindex, nodes, Tuple(I)))
end

arraysize(nodes::NTuple) = map(n->Int(maximum(n) - minimum(n) + 1), nodes)

struct NodeIterator{K,N}
    nodes::K
    iter::CartesianIndices{N}
end

"""
    iter = eachnode(ϕ)

Create an iterator for visiting all the nodes of `ϕ`.
"""
eachnode(ϕ::GridDeformation) = eachnode(ϕ.nodes)
eachnode(nodes) = NodeIterator(nodes, CartesianIndices(map(length, nodes)))

function Base.iterate(ki::NodeIterator)
    iterate(ki.iter) == nothing && return nothing
    I, state = iterate(ki.iter)
    k = node(ki.nodes, I)
    k, state
end
function Base.iterate(ki::NodeIterator, state)
    iterate(ki.iter, state) == nothing && return nothing
    I, state = iterate(ki.iter, state)
    k = node(ki.nodes, I)
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
    nodes_new = map((r,n)->range(first(r), stop=last(r), length=n), ϕ.nodes, sz)
    u = Array{SVector{N,T},N}(undef, sz)
    for (i, k) in enumerate(eachnode(nodes_new))
        u[i] = ϕ.u(k...)
    end
    GridDeformation(u, nodes_new)
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
    u, nodes = ϕ_old.u, ϕ_old.nodes
    ϕ_new.nodes == nodes || error("Not yet implemented for incommensurate nodes")
    unew = ϕ_new.u
    sz = map(length, nodes)
    x = node(nodes, 1)
    out = _compose(u, unew, x, 1)
    ucomp = similar(u, typeof(out))
    TG = similar_type(SArray, eltype(out), Size(N, N))
    g = Array{TG}(undef, size(u))
    gtmp = Vector{typeof(out)}(undef, N)
    eyeN = TG(1.0I) # eye(TG)
    for I in CartesianIndices(sz)
        x = node(nodes, I)
        dx = lookup(unew, x, I)
        y = x + dx
        ucomp[I] = dx + vecindex(u, y)
        vecgradient!(gtmp, u, y)
        g[I] = hcat(ntuple(d->gtmp[d], Val(N))...) + eyeN
    end
    GridDeformation(ucomp, nodes), g
end

"""
`ϕsi_old` and `ϕs_new` will generate `ϕs_c` vector and `g` vector
`ϕsi_old` is interpolated ``ϕs_old`:
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

"""
    ϕ = tform2deformation(tform, imgaxes, gridsize)

Construct a deformation `ϕ` from the affine transform `tform` suitable
for warping arrays with axes `imgaxes`.  The array of grid points defining `ϕ` has
size specified by `gridsize`.  The dimensionality of `tform` must
match that specified by `arraysize` and `gridsize`.

Note it's more accurate to `warp(img, tform)` directly; the main use of this function
is to initialize a GridDeformation for later optimization.
"""
function tform2deformation(tform::AffineMap{M,V},
                           imgaxes::NTuple{N,<:AbstractUnitRange},
                           gridsize::NTuple{N,<:Integer}) where {M,V,N}
    A = tform.linear - SMatrix{N,N,Float64}(I)  # this will compute the difference
    u = Array{SVector{N,eltype(M)}}(undef, gridsize...)
    nodes = map(imgaxes, gridsize) do ax, g
        range(first(ax), stop=last(ax), length=g)
    end
    for I in CartesianIndices(gridsize)
        x = SVector(map(getindex, nodes, Tuple(I)))
        u[I] = A*x + tform.translation
    end
    GridDeformation(u, nodes)
end
