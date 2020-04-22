module RegisterDeformation

using ImageCore, ImageAxes, Interpolations, StaticArrays, HDF5, JLD2, ProgressMeter
using RegisterUtilities, LinearAlgebra, Rotations, Base.Cartesian
using Distributed, Statistics, SharedArrays
using Base: tail
using Interpolations: AbstractInterpolation, AbstractExtrapolation
import ImageTransformations: warp, warp!
# to avoid `scale` conflict with Interpolations, selectively import CoordinateTransformations:
using CoordinateTransformations: AffineMap
import CoordinateTransformations: compose, transform
using OffsetArrays: IdentityUnitRange   # for Julia-version compatibility

export
    # types
    AbstractDeformation,
    GridDeformation,
    WarpedArray,
    # functions
    arraysize,       # TODO: don't export?
    centeraxes,
    compose,
    eachnode,
    extrapolate,
    extrapolate!,
    getindex!,
    griddeformations,
    interpolate,
    interpolate!,
    medfilt,
    nodegrid,
    regrid,
    similarϕ,
    tform2deformation,
    tinterpolate,
    translate,
    vecindex,        # TODO: don't export?
    vecgradient!,    # TODO: don't export?
    warp,
    warp!,
    warpgrid,
    # old AffineTransfroms.jl code (TODO?: remove)
    tformtranslate,
    tformrotate,
    tformeye,
    rotation2,
    rotation3,
    rotationparameters,
    transform!,
    transform

const DimsLike = Union{Vector{Int}, Dims}
const InterpExtrap = Union{AbstractInterpolation,AbstractExtrapolation}

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

include("griddeformation.jl")
include("utils.jl")
include("timeseries.jl")
include("tformedarrays.jl")
const Extrapolatable{T,N} = Union{TransformedArray{T,N},AbstractExtrapolation{T,N}}
include("warpedarray.jl")
include("warp.jl")
include("visualize.jl")

include("deprecated.jl")

end  # module
