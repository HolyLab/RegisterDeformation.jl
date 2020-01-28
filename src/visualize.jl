"""
    img = nodegrid(T, nodes)
    img = nodegrid(nodes)
    img = nodegrid([T], ϕ)

Returns an image `img` which has value 1 at points that are "on the
nodes."  If `nodes`/`ϕ` is two-dimensional, the image will
consist of grid lines; for three-dimensional inputs, the image will
have grid planes.

The meaning of "on the nodes" is that `x[d] == nodes[d][i]` for some
dimension `d` and node index `i`. An exception is made for the edge
values, which are brought inward by one pixel to prevent complete loss
under subpixel deformations.

Optionally specify the element type `T` of `img`.

See also [`warpgrid`](@ref).
"""
function nodegrid(::Type{T}, nodes::NTuple{N,AbstractRange}) where {T,N}
    @assert all(r->first(r)==1, nodes)
    inds = map(r->Base.OneTo(ceil(Int, last(r))), nodes)
    img = Array{T}(undef, map(length, inds))
    fill!(img, zero(T))
    for idim = 1:N
        indexes = Any[inds...]
        indexes[idim] = map(x->clamp(round(Int, x), first(inds[idim])+1, last(inds[idim])-1), nodes[idim])
        img[indexes...] .= one(T)
    end
    img
end
nodegrid(::Type{T}, ϕ::GridDeformation) where {T} = nodegrid(T, ϕ.nodes)
nodegrid(arg) = nodegrid(Bool, arg)


"""
    img = warpgrid(ϕ; [scale=1, showidentity=false])

Returns an image `img` that permits visualization of the deformation
`ϕ`.  The output is a warped rectangular grid with nodes centered on
the control points as specified by the nodes of `ϕ`.  If `ϕ` is a
two-dimensional deformation, the image will consist of grid lines; for
a three-dimensional deformation, the image will have grid planes.

`scale` multiplies `ϕ.u`, effectively making the deformation stronger
(for `scale > 1`).  This can be useful if you are trying to visualize
subtle changes. If `showidentity` is `true`, an RGB image is returned
that has the warped grid in magenta and the original grid in green.

See also [`nodegrid`](@ref).
"""
function warpgrid(ϕ; scale=1, showidentity::Bool=false)
    img = nodegrid(ϕ)
    if scale != 1
        ϕ = GridDeformation(scale*ϕ.u, ϕ.nodes)
    end
    wimg = warp(img, ϕ)
    if showidentity
        n = ndims(img)+1
        return reshape(reinterpret(RGB{Float32}, permutedims(cat(wimg, img, wimg, dims=n), (n,1:ndims(img)...))),(size(img)...,))
    end
    wimg
end
