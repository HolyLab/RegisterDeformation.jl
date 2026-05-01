export TransformedArray

"""
    A = TransformedArray(etp, tfm)

Lazy array that applies an affine coordinate transformation on access:
`A[i,j]` evaluates the parent array `etp` at the transformed coordinates
`[x, y] = tfm([i, j])`.

`etp` can be an `AbstractExtrapolation`, `AbstractInterpolation`, or plain
`AbstractArray` (automatically wrapped). `tfm` is an `AffineMap`.

See also [`transform`](@ref), [`transform!`](@ref).
"""
struct TransformedArray{T, N, E <: AbstractExtrapolation, Tf <: AffineMap} <: AbstractArray{T, N}
    data::E
    tform::Tf
end
TransformedArray(etp::AbstractExtrapolation{T, N}, a::AffineMap) where {T, N} =
    TransformedArray{T, N, typeof(etp), typeof(a)}(etp, a)

function TransformedArray(A::AbstractInterpolation, a::AffineMap)
    etp = extrapolate(A, NaN)
    return TransformedArray(etp, a)
end

function TransformedArray(A::AbstractArray, a::AffineMap)
    itp = interpolate(A, BSpline(Linear()))
    return TransformedArray(itp, a)
end


Base.size(A::TransformedArray) = size(A.data)

function Base.getindex(A::TransformedArray{T, 2}, i::Number, j::Number) where {T}
    x, y = A.tform([i, j]) #tformfwd(A.tform, i, j)
    return A.data(x, y)
end

function Base.getindex(A::TransformedArray{T, 3}, i::Number, j::Number, k::Number) where {T}
    x, y, z = A.tform([i, j, k]) #tformfwd(A.tform, i, j, k)
    return A.data(x, y, z)
end

(A::TransformedArray)(args::Number...) = A[args...]

Base.similar(A::TransformedArray, ::Type{T}, dims::Dims) where {T} = Array{T}(undef, dims)

"""
    transform(A::TransformedArray; origin_dest=center(A), origin_src=center(A)) -> Array
    transform(A, tfm::AffineMap; origin_dest=center(A), origin_src=center(A)) -> Array

Materialize the transformed array `A` over its entire domain. By default the
transformation is assumed to operate around the center of the input array, and
output coordinates are referenced relative to the center of the output.

The two-argument form wraps `A` in a `TransformedArray` first. The `getindex`
behavior (`A[i,j]`) assumes the origin at zero; to match it, pass
`origin_src = zeros(N)` and `origin_dest = zeros(N)`, or equivalently offset
the transform by `origin_src - tfm.linear * origin_dest`.

See also [`transform!`](@ref).
"""
function transform(A::TransformedArray{T, N}; kwargs...) where {T, N}
    y = A.tform(ones(Int, N))
    yt = (y...,)::NTuple{N, eltype(y)}
    a = A.data(yt...)
    dest = Array{typeof(a)}(undef, size(A))
    transform!(dest, A; kwargs...)
    return dest
end

transform(A, a::AffineMap; kwargs...) = transform(TransformedArray(A, a); kwargs...)

"""
    transform!(dest, src::TransformedArray; origin_dest=center(dest), origin_src=center(src)) -> dest
    transform!(dest, src, tfm::AffineMap; origin_dest=center(dest), origin_src=center(src)) -> dest

In-place version of [`transform`](@ref). Writes the result into the
pre-allocated array `dest` and returns it.
"""
function transform!(
        dest::AbstractArray{S, N},
        src::TransformedArray{T, N};
        origin_dest = center(dest),
        origin_src = center(src)
    ) where {S, T, N}
    tform = src.tform
    if tform.linear == Matrix{T}(I, N, N) && tform.translation == zeros(N) && size(dest) == size(src) && origin_dest == origin_src
        copyto!(dest, src)
        return dest
    end
    offset = tform.translation - tform.linear * origin_dest + origin_src
    return _transform!(dest, src, offset)
end

transform!(dest, src, a::AffineMap; kwargs...) = transform!(dest, TransformedArray(src, a); kwargs...)

@require ImageMetadata = "bc367c6b-8a6b-528e-b4bd-a4b897500b49" begin
    transform(A::ImageMetadata.ImageMeta, a::AffineMap; kwargs...) = ImageMetadata.copyproperties(A, transform(ImageMetadata.data(A), a; kwargs...))
    transform!(dest, A::ImageMetadata.ImageMeta, a::AffineMap; kwargs...) = ImageMetadata.copyproperties(A, transform!(dest, ImageMetadata.data(A), a; kwargs...))
end

# For a FilledExtrapolation, this is designed to (usually) avoid evaluating
# the interpolation unless it is in-bounds.  This often improves performance.
@generated function _transform!(
        dest::AbstractArray{S, N},
        src::TransformedArray{T, N, E},
        offset
    ) where {S, T, N, E <: Interpolations.FilledExtrapolation}
    # Initialize the final column of s "matrix," e.g., s_1_3 = A_1_3 + o_3
    # s stands for source-coordinates. The first "column," s_d_1, corresponds
    # to the actual interpolation position. The later columns simply cache
    # previous computations.
    sN = ntuple(i -> Expr(:(=), Symbol(string("s_", i, "_$N")), Expr(:call, :+, Symbol(string("A_", i, "_$N")), Symbol(string("o_", i)))), N)
    return quote
        tform = src.tform
        data = src.data
        @nexprs $N d -> (o_d = offset[d])
        @nexprs $N j -> (@nexprs $N i -> (A_i_j = tform.linear[i, j]))
        fill!(dest, data.fillvalue)
        $(sN...)
        @nloops(
            $N, i, d -> (d > 1 ? (1:size(dest, d)) : (imin:imax)),
            # The pre-expression chooses the range within each column that will be in-bounds
            d -> (
                d > 2 ? (@nexprs $N e -> s_e_{d - 1} = A_e_{d - 1} + s_e_d) :
                    d == 2 ? begin
                        imin = 1
                        imax = size(dest, 1)
                        @nexprs $N e -> ((imin, imax) = irange(imin, imax, A_e_1, s_e_2, size(data, e)))
                        @nexprs $N e -> (s_e_1 = A_e_1 * imin + s_e_d)
                    end :
                    nothing
            ), # pre
            d -> (@nexprs $N e -> (s_e_d += A_e_d)), # post
            # Perform the interpolation of the source data
            @inbounds (@nref $N dest i) = (@ncall $N data d -> s_d_1)
        )
        dest
    end
end

center(A::AbstractArray) = [(size(A, d) + 1) / 2 for d in 1:ndims(A)]

# Find i such that
#      imin <= i <= imax
#      1 <= coef*i+offset <= upper
function irange(imin::Int, imax::Int, coef, offset, upper)
    thresh = 10^4 / typemax(Int)  # needed to avoid InexactError for results with abs() bigger than typemax
    if coef > thresh
        return max(imin, floor(Int, (1 - offset) / coef)), min(imax, ceil(Int, (upper - offset) / coef))
    elseif coef < -thresh
        return max(imin, floor(Int, (upper - offset) / coef)), min(imax, ceil(Int, (1 - offset) / coef))
    else
        if 1 <= offset <= upper
            return imin, imax
        else
            return 1, 0   # empty range
        end
    end
end
