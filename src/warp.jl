"""
    warp(img, ϕ) -> Array

Warp the array `img` according to the deformation `ϕ`. Returns an array of the
same size and axes as `img`.
"""
function warp(img::AbstractArray, ϕ::AbstractDeformation)
    wimg = WarpedArray(img, ϕ)
    dest = similar(img, warp_type(img))
    return warp!(dest, wimg)
end

warp_type(img::AbstractArray{T}) where {T <: AbstractFloat} = T
warp_type(img::AbstractArray{T}) where {T <: Number} = Float32
warp_type(img::AbstractArray{C}) where {C <: Colorant} = warp_type(img, eltype(eltype(C)))
warp_type(img::AbstractArray{C}, ::Type{T}) where {C <: Colorant, T <: AbstractFloat} = C
warp_type(img::AbstractArray{C}, ::Type{T}) where {C <: Colorant, T} = base_colorant_type(C){Float32}

"""
    warp!(dest, src::WarpedArray) -> dest

Instantiate the `WarpedArray` `src` into the pre-allocated array `dest`. Returns `dest`.
"""
function warp!(dest::AbstractArray{T, N}, src::WarpedArray) where {T, N}
    axes(dest) == axes(src) || throw(DimensionMismatch("dest must have the same axes as src"))
    destiter = CartesianIndices(axes(dest))
    I, deststate = iterate(destiter)
    for ux in eachvalue(src.ϕ.u)
        dest[I] = src.data((Tuple(I) .+ ux)...)
        if deststate == last(destiter)
            break
        end
        I, deststate = iterate(destiter, deststate)
    end
    return dest
end

"""
    warp!(dest, img, ϕ) -> dest

Warp `img` using the deformation `ϕ`, storing the result in the pre-allocated
array `dest`. Returns `dest`.
"""
function warp!(dest::AbstractArray, img::AbstractArray, ϕ::AbstractDeformation)
    wimg = WarpedArray(to_etp(img), ϕ)
    return warp!(dest, wimg)
end

"""
    warp!(dest, img, tform, ϕ) -> dest

Warp `img` by applying the affine transformation `tform` followed by the deformation
`ϕ`, storing the result in the pre-allocated array `dest`. Returns `dest`.
"""
function warp!(dest::AbstractArray, img::AbstractArray, A::AffineMap, ϕ::AbstractDeformation)
    wimg = WarpedArray(to_etp(img, A), ϕ)
    return warp!(dest, wimg)
end

"""
    warp!(io, img, ϕs; eltype=Float32, nworkers=1)
    warp!(io, img, uarray; eltype=Float32, nworkers=1)

Write warped images to disk. `io` is an `IO` object or a pre-allocated HDF5/JLD2
dataset. `img` is an image sequence and `ϕs` is a vector of deformations, one per
image in `img`. `eltype` controls the element type written to disk. If `nworkers > 1`,
additional worker processes are spawned to parallelize the deformation.

In the second form, `uarray` is an array of displacement values with
`size(uarray)[end] == nimages(img)`.
"""
function warp!(dest::Union{IO, HDF5.Dataset, JLD2.JLDFile}, img, ϕs; eltype::Type = Float32, nworkers = 1)
    T = eltype
    n = nimages(img)
    saxs = indices_spatial(img)
    ssz = map(length, saxs)
    if n == 1
        ϕ = extract1(ϕs, sdims(img), saxs)
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
    @showprogress dt = 1 desc = "Stacks:" for i in 1:n
        ϕ = extracti(ϕs, i, saxs)
        warp!(destarray, view(img, timeaxis(img)(i)), ϕ)
        warp_write(dest, destarray, i)
    end
    return nothing
end

warp!(dest::Union{IO, HDF5.Dataset, JLD2.JLDFile}, img, u::AbstractArray{<:Real}; kwargs...) =
    warp!(dest, img, Array(convert_to_fixed(u)); kwargs...)

warp!(dest::Union{HDF5.Dataset, JLD2.JLDFile}, img, u; nworkers = 1) =
    warp!(dest, img, u; eltype = Base.eltype(dest), nworkers)

# Disambiguate with HDF5/JLD2 dest + AbstractArray{<:Real} u (fixes ambiguity between the two methods above)
warp!(dest::Union{HDF5.Dataset, JLD2.JLDFile}, img, u::AbstractArray{<:Real}; nworkers = 1) =
    warp!(dest, img, Array(convert_to_fixed(u)); eltype = Base.eltype(dest), nworkers)

# Disambiguate with ImageTransformations.warp(::AbstractExtrapolation, tform)
function warp(img::Interpolations.AbstractExtrapolation, ϕ::AbstractDeformation)
    wimg = WarpedArray(img, ϕ)
    dest = similar(parent(img), warp_type(img))
    return warp!(dest, wimg)
end

# Disambiguate with ImageTransformations.warp!(out, ::AbstractExtrapolation, tform)
function warp!(dest::AbstractArray, img::Interpolations.AbstractExtrapolation, ϕ::AbstractDeformation)
    wimg = WarpedArray(img, ϕ)
    return warp!(dest, wimg)
end

# Disambiguate AbstractExtrapolation img with IO/HDF5/JLD2 dest variants
warp!(dest::Union{HDF5.Dataset, JLD2.JLDFile}, img::Interpolations.AbstractExtrapolation, u; nworkers = 1) =
    warp!(dest, img, u; eltype = Base.eltype(dest), nworkers)

warp!(dest::Union{HDF5.Dataset, JLD2.JLDFile}, img::Interpolations.AbstractExtrapolation, u::AbstractArray{<:Real}; nworkers = 1) =
    warp!(dest, img, Array(convert_to_fixed(u)); eltype = Base.eltype(dest), nworkers)

function warp!(
        dest::Union{IO, HDF5.Dataset, JLD2.JLDFile},
        img::Interpolations.AbstractExtrapolation,
        ϕs;
        eltype::Type = Float32,
        nworkers = 1,
    )
    return invoke(
        warp!,
        Tuple{Union{IO, HDF5.Dataset, JLD2.JLDFile}, Any, Any},
        dest,
        img,
        ϕs;
        eltype,
        nworkers,
    )
end

warp!(
    dest::Union{IO, HDF5.Dataset, JLD2.JLDFile},
    img::Interpolations.AbstractExtrapolation,
    u::AbstractArray{<:Real};
    kwargs...,
) = warp!(dest, img, Array(convert_to_fixed(u)); kwargs...)

function _warp!(::Type{T}, dest, img, ϕs, nworkers) where {T}
    n = nimages(img)
    saxs = indices_spatial(img)
    ssz = map(length, saxs)
    wpids = addprocs(nworkers)
    simg = Vector{Any}()
    swarped = Vector{Any}()
    rrs = Vector{RemoteChannel}()
    mydir = splitdir(@__FILE__)[1]
    pkgbase = String(chop(mydir, tail = 4))
    for p in wpids
        remotecall_fetch(Main.eval, p, :(using Pkg))
        remotecall_fetch(Main.eval, p, :(Pkg.activate($pkgbase)))
        remotecall_fetch(Main.eval, p, :(push!(LOAD_PATH, $mydir)))
        remotecall_fetch(Main.eval, p, :(using RegisterDeformation))
        push!(simg, SharedArray{eltype(img)}(ssz, pids = [myid(), p]))
        push!(swarped, SharedArray{T}(ssz, pids = [myid(), p]))
    end
    nextidx = 0
    getnextidx() = nextidx += 1
    writing_mutex = RemoteChannel()
    prog = Progress(n; dt = 1, desc = "Stacks:")
    @sync begin
        for i in 1:nworkers
            p = wpids[i]
            src = simg[i]
            warped = swarped[i]
            @async begin
                while (idx = getnextidx()) <= n
                    ϕ = extracti(ϕs, idx, saxs)
                    copyto!(src, view(img, timeaxis(img)(idx)))
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
    return nothing
end

warp_write(io::IO, destarray) = write(io, destarray)
function warp_write(io::IO, destarray, i)
    offset = (i - 1) * length(destarray) * sizeof(eltype(destarray))
    seek(io, offset)
    return write(io, destarray)
end
function warp_write(dest, destarray, i)
    colons = [Colon() for d in 1:ndims(destarray)]
    return dest[colons..., i] = destarray
end

"""
    translate(A, displacement) -> Array

Shift `A` by `displacement` applied to the spatial coordinates. In simple cases,
`result[i, j, ...] = A[i+displacement[1], j+displacement[2], ...]`. Missing pixels
are filled with `NaN`.
"""
function translate(A::AbstractArray, displacement::Union{AbstractVector{<:Integer}, Dims})
    disp = zeros(Int, ndims(A))
    disp[[coords_spatial(A)...]] = displacement
    indx = UnitRange{Int}[ axes(A, i) .+ disp[i] for i in 1:ndims(A) ]
    return get(A, indx, NaN)
end
