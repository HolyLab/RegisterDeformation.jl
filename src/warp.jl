
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
function warp!(dest::AbstractArray{T,N}, src::WarpedArray) where {T,N}
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
function warp!(::Type{T}, dest::Union{IO,HDF5.Dataset,JLD2.JLDFile}, img, ϕs; nworkers=1) where T
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
    @showprogress dt=1 desc="Stacks:" for i = 1:n
        ϕ = extracti(ϕs, i, saxs)
        warp!(destarray, view(img, timeaxis(img)(i)), ϕ)
        warp_write(dest, destarray, i)
    end
    nothing
end

warp!(::Type{T}, dest::Union{IO,HDF5.Dataset,JLD2.JLDFile}, img, u::AbstractArray{R}; kwargs...) where {T,R<:Real} = warp!(T, dest, img, Array(convert_to_fixed(u)); kwargs...)

warp!(dest::Union{HDF5.Dataset,JLD2.JLDFile}, img, u; nworkers=1) =
    warp!(eltype(dest), dest, img, u; nworkers=nworkers)

function _warp!(::Type{T}, dest, img, ϕs, nworkers) where T
    n = nimages(img)
    saxs = indices_spatial(img)
    ssz = map(length, saxs)
    wpids = addprocs(nworkers)
    simg = Vector{Any}()
    swarped = Vector{Any}()
    rrs = Vector{RemoteChannel}()
    mydir = splitdir(@__FILE__)[1]
    pkgbase = String(chop(mydir,tail=4))
    for p in wpids
        remotecall_fetch(Main.eval, p, :(using Pkg))
        remotecall_fetch(Main.eval, p, :(Pkg.activate($pkgbase)))
        remotecall_fetch(Main.eval, p, :(push!(LOAD_PATH, $mydir)))
        remotecall_fetch(Main.eval, p, :(using RegisterDeformation))
        push!(simg, SharedArray{eltype(img)}(ssz, pids=[myid(),p]))
        push!(swarped, SharedArray{T}(ssz, pids=[myid(),p]))
    end
    nextidx = 0
    getnextidx() = nextidx += 1
    writing_mutex = RemoteChannel()
    prog = Progress(n; dt=1, desc="Stacks:")
    @sync begin
        for i = 1:nworkers
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

"""
`Atrans = translate(A, displacement)` shifts `A` by an amount
specified by `displacement`.  Specifically, in simple cases `Atrans[i,
j, ...] = A[i+displacement[1], j+displacement[2], ...]`.  More
generally, `displacement` is applied only to the spatial coordinates
of `A`.

`NaN` is filled in for any missing pixels.
"""
function translate(A::AbstractArray, displacement::DimsLike)
    disp = zeros(Int, ndims(A))
    disp[[coords_spatial(A)...]] = displacement
    indx = UnitRange{Int}[ axes(A, i) .+ disp[i] for i = 1:ndims(A) ]
    get(A, indx, NaN)
end
