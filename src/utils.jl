centeraxes(axs) = map(centeraxis, axs)

function centeraxis(ax)
    f, l = first(ax), last(ax)
    n = l - f + 1
    nhalf = (n+1) ÷ 2
    return 1-nhalf:n-nhalf
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


# TODO?: do we need to return real values beyond-the-edge for a SubArray?
to_etp(img) = extrapolate(interpolate(img, BSpline(Linear())), convert(promote_type(eltype(img), Float32), NaN))

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
    reshape(reinterpret(SVector{N,T}, vec(u)), sz)
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

# Wrapping functions to interface with CoordinateTransfromations instead of AffineTransfroms module
tformeye(m::Int) = AffineMap(Matrix{Float64}(I,m,m), zeros(m))
tformtranslate(trans::Vector) = (m = length(trans); AffineMap(Matrix{Float64}(I,m,m), trans))

rotation2(angle) = RotMatrix(angle)
function tformrotate(angle)
    A = RotMatrix(angle)
    AffineMap(A, zeros(eltype(A),2))
end

function rotationparameters(R::Matrix)
    size(R, 1) == size(R, 2) || error("Matrix must be square")
    if size(R, 1) == 2
        return [atan(-R[1,2],R[1,1])]
    elseif size(R, 1) == 3
        aa = AngleAxis(R)
        return rotation_angle(aa)*rotation_axis(aa)
    else
        error("Rotations in $(size(R, 1)) dimensions not supported")
    end
end

function rotation3(axis::Vector{T}, angle) where T
    n = norm(axis)
    axisn = n>0 ? axis/n : (tmp = zeros(T,length(axis)); tmp[1] = 1; tmp)
    AngleAxis(angle, axisn...)
end

function rotation3(axis::Vector{T}) where T
    n = norm(axis)
    axisn = n>0 ? axis/n : (tmp = zeros(typeof(one(T)/1),length(axis)); tmp[1] = 1; tmp)
    AngleAxis(n, axisn...)
end


function tformrotate(axis::Vector, angle)
    if length(axis) == 3
        return AffineMap(rotation3(axis, angle), zeros(eltype(axis),3))
    else
        error("Dimensionality ", length(axis), " not supported")
    end
end

function tformrotate(x::Vector)
    if length(x) == 3
        return AffineMap(rotation3(x), zeros(eltype(x),3))
    else
        error("Dimensionality ", length(x), " not supported")
    end
end


#=
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
