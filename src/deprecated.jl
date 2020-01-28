@inline function Base.getproperty(ϕ::GridDeformation, name::Symbol)
    if name === :u
        return getfield(ϕ, :u)
    elseif name === :nodes
        return getfield(ϕ, :nodes)
    elseif name === :knots
        Base.depwarn("the field name of `GridDeformation` has changed from `knots` to `nodes`.", :getproperty)
        return getfield(ϕ, :nodes)
    end
    error("field ", name, " is not available")
end

function GridDeformation(u::AbstractArray{FV,N},
                         sz::NTuple{N,L}) where {FV<:SVector,N,L<:Integer}
    Base.depwarn("`GridDeformation(u, size(fixed))` is deprecated, use `GridDeformation(u, axes(fixed))` instead.", :GridDeformation)
    return GridDeformation(u, map(Base.OneTo, sz))
end

# The above causes an ambiguity, resolve it
function GridDeformation(u::AbstractArray{FV,0}, nodes::Tuple{}) where FV<:SVector
    error("node tuple cannot be empty")
end

function GridDeformation(u, nodes::AbstractVector{<:Integer})
    Base.depwarn("`GridDeformation(u, [size(fixed)...])` is deprecated, pass the axes of `fixed` instead.", :GridDeformation)
    GridDeformation(u, map(Base.OneTo, (nodes...,)))
end

function tform2deformation(tform::AffineMap{M,V}, arraysize::DimsLike, gridsize) where {M,V}
    Base.depwarn("""
    `tform2deformation(tform, arraysize, gridsize)` is deprecated.
    Formerly, `tform` was defined so as to apply to `centered(img)`, which shifts
    the axes of `img` to put 0 at the midpoint of `img`.
    Now you should call this as `tform2deformation(tform, axs, gridsize)`, where
    `axs` represents the desired domain of `tform`.
    If you are transitioning old code, either use `axs = axes(fixed)` if `fixed`
    is already centered, or `axs = centeraxes(axes(fixed))` if not.
    """, :tform2deformation)
    axs = map((arraysize...,)) do sz
        halfsz = sz ÷ 2
        IdentityUnitRange(1-halfsz:sz-halfsz)
    end
    return tform2deformation(tform, axs, (gridsize...,))
end

import Base: getindex
@deprecate getindex(ϕ::GridDeformation{T,N,A}, xs::Vararg{Number,N}) where {T,N,A<:AbstractInterpolation} ϕ(xs...)

Base.@deprecate_binding eachknot eachnode
Base.@deprecate_binding knotgrid nodegrid
