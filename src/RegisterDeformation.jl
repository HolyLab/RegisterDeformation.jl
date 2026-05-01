module RegisterDeformation

using CoordinateTransformations: CoordinateTransformations, AffineMap
using Distributed: Distributed, RemoteChannel, addprocs, myid, nworkers, remotecall_fetch
using HDF5: HDF5
using ImageAxes: ImageAxes, AbstractGray, AbstractRGB, Colorant, Gray, RGB,
    base_colorant_type, coords_spatial, data, getindex!, indices_spatial, nimages, sdims,
    timeaxis
using Interpolations: Interpolations, AbstractExtrapolation, AbstractInterpolation, BSpline,
    Flat, InPlace, Line, Linear, NoInterp, OnCell, Quadratic, ScaledInterpolation,
    eachvalue, extrapolate, interpolate, interpolate!, scale
using JLD2: JLD2
using LinearAlgebra: LinearAlgebra, I, norm
using ProgressMeter: ProgressMeter, Progress, finish!, update!, @showprogress
using RegisterUtilities: RegisterUtilities
using Requires: Requires, @require
using Rotations: Rotations, AngleAxis, RotMatrix, rotation_angle, rotation_axis
using SharedArrays: SharedArrays, SharedArray
using StaticArrays: StaticArrays, SArray, SVector, Size, similar_type
using Statistics: Statistics, median!
import CoordinateTransformations: compose
import ImageTransformations: warp, warp!
using Base: tail
using Base.Cartesian

export
    # types
    AbstractDeformation,
    GridDeformation,
    NodeIterator,
    WarpedArray,
    # functions
    arraysize, # TODO: don't export?
    centeraxes,
    compose,
    eachnode,
    extrapolate,
    extrapolate!,
    getindex!,
    griddeformations,
    interpolate,
    interpolate!,
    nodegrid,
    regrid,
    similar_deformation,
    tform2deformation,
    tinterpolate,
    tmedfilt,
    tmedfilt!,
    translate,
    vecindex, # TODO: don't export?
    vecgradient!, # TODO: don't export?
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

const InterpExtrap = Union{AbstractInterpolation, AbstractExtrapolation}

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

"""
    AbstractDeformation{T,N}

Supertype for N-dimensional deformations with displacement element type `T`.

A deformation maps a point `x` to `x + u(x)`, where `u` is the displacement
field. The concrete type `GridDeformation` represents `u` on a regular grid
with interpolation between grid points.

Use `eltype(ϕ)` and `ndims(ϕ)` to query the element type and dimensionality.
"""
abstract type AbstractDeformation{T, N} end
Base.eltype(::Type{AbstractDeformation{T, N}}) where {T, N} = T
Base.ndims(::Type{AbstractDeformation{T, N}}) where {T, N} = N
Base.eltype(::Type{D}) where {D <: AbstractDeformation} = eltype(supertype(D))
Base.ndims(::Type{D}) where {D <: AbstractDeformation} = ndims(supertype(D))
Base.eltype(d::AbstractDeformation) = eltype(typeof(d))
Base.ndims(d::AbstractDeformation) = ndims(typeof(d))

include("griddeformation.jl")
include("utils.jl")
include("timeseries.jl")
include("tformedarrays.jl")
const Extrapolatable{T, N} = Union{TransformedArray{T, N}, AbstractExtrapolation{T, N}}
include("warpedarray.jl")
include("warp.jl")
include("visualize.jl")

end  # module
