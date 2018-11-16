#import BlockRegistration
import RegisterDeformation
using CoordinateTransformations, Interpolations, ColorTypes, ForwardDiff
using StaticArrays, Images, AxisArrays, LinearAlgebra, Distributed, Statistics
using Test

using RegisterUtilities

znan(x) = isnan(x) ? zero(x) : x

knots = (range(1, stop=15, length=5), range(1, stop=11, length=3))
@test RegisterDeformation.arraysize(knots) == (15,11)

knotsall = (knots[1].*ones(length(knots[2]))', ones(length(knots[1])).*knots[2]')
for (i, knt) in enumerate(RegisterDeformation.eachknot(knots))
    @test knt[1] == knotsall[1][i]
    @test knt[2] == knotsall[2][i]
end

randrange(xmin, xmax) = (xmax-xmin)*rand() + xmin

## Deformations
for knts in ((range(1, stop=15, length=5),),
              (range(1, stop=15, length=5), range(1, stop=8, length=4)),
              (range(1, stop=15, length=3), range(1, stop=8, length=4), range(1, stop=7, length=2)),
              (range(1, stop=15, length=5), range(1, stop=8, length=4), range(1, stop=7, length=6)))
    sz = map(length, knts)
    global rng = map(z->convert(Int, maximum(z)), knts)
    # Zero deformation
    global u = zeros(length(knts), sz...)
    global ϕ = interpolate(RegisterDeformation.GridDeformation(u, knts))
    for i = 1:10
        x = map(z->rand(1:z), rng)
        @test ϕ[x...] ≈ [x...]
        if all(x->x>2, sz)
            y = map(z->rand(2:z-1)+rand()-0.5, rng)
            @test ϕ[y...] ≈ [y...]
        end
    end
    # Constant shift
    dx = randn(length(knts))
    global u = zeros(length(knts), sz...).+dx
    global ϕ = interpolate(RegisterDeformation.GridDeformation(u, knts))
    for i = 1:10
        x = map(z->rand(1:z), rng)
        @test ϕ[x...] ≈ [x...]+dx
        if all(x->x>2, sz)
            y = map(z->rand(2:z-1)+rand()-0.5, rng)
            @test ϕ[y...] ≈ [y...]+dx
        end
    end
    # "Biquadratic"
    if all(x->x>2, sz)
        N = length(knts)
        u_is = Vector{Any}(undef, N)
        fs = Vector{Any}(undef, N)
        R = CartesianIndices(sz)
        for i = 1:N
            let cntr = Float64[rand(1:z-1)+rand() for z in rng], q = rand(N)
                f = x -> dot(q, (x-cntr).^2)
                ui = zeros(sz)
                for I in R
                    ui[I] = f(Float64[knts[d][I[d]] for d = 1:N])
                end
                u_is[i] = reshape(ui, 1, sz...)
                fs[i] = f
            end
        end
        global u = cat(u_is..., dims = 1)
        global ϕ = interpolate!(RegisterDeformation.GridDeformation(copy(u), knts), InPlaceQ(OnCell()))
        for i = 1:10
            y = Float64[randrange((knot[2]+knot[1])/2, (knot[end]+knot[end-1])/2) for knot in knts]
            dx = Float64[fs[d](y) for d=1:N]
            try
                @test ϕ[y...] ≈ y + Float64[fs[d](y) for d=1:N]
            catch err
                @show y knts
                @show [((knot[2]+knot[1])/2, (knot[end]+knot[end-1])/2) for knot in knts]
                @show Float64[fs[d](y) for d=1:N]
                rethrow(err)
            end
        end
    end
end


s = [3.3,-2.6]
gsize = (3,2)
A = RegisterDeformation.tformtranslate(s)
ϕ = RegisterDeformation.tform2deformation(A, (500,480), gsize)
u = reshape(reinterpret(Float64, ϕ.u), tuple(2,gsize...))
@test all(u[1,:,:] .== s[1])
@test all(u[2,:,:] .== s[2])

## WarpedArray

p = (1:100).-5

dest = Vector{Float32}(undef, 10)

# Simple translations in 1d
ϕ = RegisterDeformation.GridDeformation([0.0,0.0]', size(p))
q = RegisterDeformation.WarpedArray(p, ϕ);
RegisterDeformation.getindex!(dest, q, 1:10)
@assert maximum(abs.(dest - p[1:10])) < 10*eps(maximum(abs.(dest)))

ϕ = RegisterDeformation.GridDeformation([1.0,1.0]', size(p))
q = RegisterDeformation.WarpedArray(p, ϕ);
RegisterDeformation.getindex!(dest, q, 1:10)
@assert maximum(abs.(dest - p[2:11])) < 10*eps(maximum(abs.(dest)))

ϕ = RegisterDeformation.GridDeformation([-2.0,-2.0]', size(p))
q = RegisterDeformation.WarpedArray(p, ϕ);
RegisterDeformation.getindex!(dest, q, 1:10)
@assert maximum(abs.(dest[3:end] - p[1:8])) < 10*eps(maximum(abs.(dest[3:end])))

RegisterDeformation.getindex!(dest, q, 3:12)
@assert maximum(abs.(dest - p[1:10])) < 10*eps(maximum(abs.(dest)))

ϕ = RegisterDeformation.GridDeformation([2.0,2.0]', size(p))
q = RegisterDeformation.WarpedArray(p, ϕ);
RegisterDeformation.getindex!(dest, q, 1:10)
@assert maximum(abs.(dest - p[3:12])) < 10*eps(maximum(abs.(dest)))

ϕ = RegisterDeformation.GridDeformation([5.0,5.0]', size(p))
q = RegisterDeformation.WarpedArray(p, ϕ);
RegisterDeformation.getindex!(dest, q, 1:10)
@assert maximum(abs.(dest - p[6:15])) < 10*eps(maximum(abs.(dest)))

# SubArray (test whether we can go beyond edges)
psub = view(collect(p), 3:20)
ϕ = RegisterDeformation.GridDeformation([0.0,0.0]', size(p))
q = RegisterDeformation.WarpedArray(psub, ϕ);
RegisterDeformation.getindex!(dest, q, -1:8)
@assert maximum(abs.(dest[3:end] - p[3:10])) < 10*eps(maximum(abs.(dest[3:end])))
any(isnan, dest) && @warn("Some dest are NaN, not yet sure whether this is a problem")

# Stretches
u = [0.0,5.0,10.0]
ϕ = interpolate(RegisterDeformation.GridDeformation(u', size(p)), Line(OnCell()))
q = RegisterDeformation.WarpedArray(p, ϕ)
RegisterDeformation.getindex!(dest, q, 1:10)
@assert abs(dest[1] - p[1]) < sqrt(eps(1.0f0))
RegisterDeformation.getindex!(dest, q, 86:95)
@assert isnan.(dest) == [falses(5);trues(5)]  # fixme
dest2 = RegisterDeformation.getindex!(zeros(Float32, 100), q, 1:100)
@assert all(abs.(diff(dest2)[26:74] .- ((u[3]-u[1])/99+1)) .< sqrt(eps(1.0f0)))

#2d
p = reshape(1:120, 10, 12)
u1 = [2.0 2.0; 2.0 2.0]
u2 = [-1.0 -1.0; -1.0 -1.0]
ϕ = RegisterDeformation.GridDeformation((u1,u2), size(p))
q = RegisterDeformation.WarpedArray(p, ϕ)
dest = zeros(size(p))
rng = (1:size(p,1),1:size(p,2))
RegisterDeformation.getindex!(dest, q, rng...)
@assert maximum(abs.(znan.(dest[1:7,2:end] - p[3:9,1:end-1]))) < 10*eps(maximum(znan.(dest)))

# Composition
# Test that two rotations approximately compose to another rotation
gridsize = (49,47)
imgsize = (200,200)
ϕ1 = RegisterDeformation.tform2deformation(RegisterDeformation.tformrotate( pi/180), imgsize, gridsize)
ϕ2 = RegisterDeformation.tform2deformation(RegisterDeformation.tformrotate(2pi/180), imgsize, gridsize)
ϕi = interpolate(ϕ1)
ϕc = ϕi(ϕi)
uc = reshape(reinterpret(Float64, ϕc.u), (2, gridsize...))
u2 = reshape(reinterpret(Float64, ϕ2.u), (2, gridsize...))
@test ≈(uc, u2, rtol=0.02, norm=x->maximum(abs, x))

## Test gradient computation

function compare_g(g, gj, gridsize)
    for j = 1:gridsize[2], i = 1:gridsize[1]
        indx = (LinearIndices(gridsize))[i,j]
        global rng = 2*(indx-1)+1:2indx
        @test g[i,j] ≈ gj[rng, rng]
        for k = 1:2*prod(gridsize)
            if !in(k, rng)
                @test abs(gj[k,rng[1]]) < 1e-14
                @test abs(gj[k,rng[2]]) < 1e-14
            end
        end
    end
    nothing
end

# A composition function that ForwardDiff will be happy with
function compose_u(ϕ1, u2, f, shp, imsz)
    ϕ2 = f(RegisterDeformation.GridDeformation(reshape(u2, shp), imsz))
    ϕ = ϕ1(ϕ2)
    ret = Vector{eltype(eltype(ϕ.u))}(undef, prod(shp))
    i = 0
    for v in ϕ.u
        for d = 1:length(v)
            ret[i+=1] = v[d]
        end
    end
    ret
end

gridsize = (11,11)
u1 = randn(2, gridsize...)
u2 = randn(2, gridsize...)
imsz = (100,101)
ϕ1 = interpolate(RegisterDeformation.GridDeformation(u1, imsz))
for f in (interpolate, identity)
    global ϕ2 = f(RegisterDeformation.GridDeformation(u2, imsz))
    global ϕ, g = RegisterDeformation.compose(ϕ1, ϕ2)
    u2vec = vec(u2)
    global gj = ForwardDiff.jacobian(u2vec -> compose_u(ϕ1, u2vec, f, size(u2), imsz), u2vec)
    compare_g(g, gj, gridsize)
end

# Test identity-composition
_, g = RegisterDeformation.compose(identity, ϕ2)
gj = zeros(2*prod(gridsize), 2*prod(gridsize))
for i = 1:prod(gridsize)
    global rng = 2*(i-1)+1:2i
    gj[rng,rng] = Matrix{Float64}(I,2,2)
end
compare_g(g, gj, gridsize)

# warpgrid
knots = (range(1, stop=100, length=4), range(1, stop=100, length=3))
gridsize = map(length, knots)
ϕ = RegisterDeformation.GridDeformation(5*randn(2,gridsize...), knots)
kg = RegisterDeformation.knotgrid(knots)
@test eltype(kg) == Bool
@test size(kg) == (100, 100)
kg2 = RegisterDeformation.knotgrid(ϕ)
@test kg2 == kg
kg = RegisterDeformation.knotgrid(Float64, knots)
@test eltype(kg) == Float64
A = RegisterDeformation.warpgrid(ϕ)
B = RegisterDeformation.warpgrid(ϕ, scale=1.5)
@test A != B
@test isa(A, Matrix{Float32})
C = RegisterDeformation.warpgrid(ϕ, showidentity=true)
@test eltype(C) == RGB{Float32}

# Writing warped data to disk
o = ones(Float32, 5, 5)
A = o .* reshape(1:7, (1,1,7))
img = AxisArray(A, :y, :x, :time)
fn = tempname()
# With Vector{GridDeformation}
ϕs = tighten([RegisterDeformation.GridDeformation(zeros(2,3,3), size(o)) for i = 1:nimages(img)])
open(fn, "w") do io
    RegisterDeformation.warp!(Float32, io, img, ϕs)
end
warped = open(fn, "r") do io
    read!(io, Array{Float32}(undef, size(img)))
end
@test warped == img
# With Array{SVector}
uarray = reshape(reinterpret(SVector{2,Float64}, zeros(2,3,3,7)), (3,3,7))
open(fn, "w") do io
    RegisterDeformation.warp!(Float32, io, img, uarray)
end
warped = open(fn, "r") do io
    read!(io, Array{Float32}(undef, size(img)))
end
@test warped == img
# With Array{Real}
uarray = zeros(2,3,3,7)
fn = tempname()
open(fn, "w") do io
    RegisterDeformation.warp!(Float32, io, img, uarray)
end
warped = open(fn, "r") do io
    read!(io, Array{Float32}(undef, size(img)))
end
@test warped == img
# Multi-process
open(fn, "w") do io
    RegisterDeformation.warp!(Float32, io, img, uarray; nworkers=3)
end
warped = open(fn, "r") do io
    read!(io, Array{Float32}(undef, size(img)))
end
@test warped == img

# Saving arrays-of-fixedsizearrays efficiently
using JLD2
knots = (range(1, stop=20, length=3), range(1, stop=30, length=5))
ϕ = RegisterDeformation.GridDeformation(rand(2,map(length,knots)...), knots)
fn = string(tempname(), ".jld")
save(fn, "ϕ", ϕ)
ϕ2 = load(fn, "ϕ")
@test ϕ2.u == ϕ.u
@test ϕ2.knots == ϕ.knots
str = read(`h5dump $fn`, String)
@test something(findfirst("SIMPLE { ( 5, 3, 2 ) / ( 5, 3, 2 ) }", str), 0:-1) != 0:-1||
      something(findfirst("SIMPLE { ( 30 ) / ( 30 ) }", str), 0:-1) != 0:-1||
      something(findfirst("SIMPLE { ( 2 ) / ( 2 ) }", str), 0:-1) != 0:-1
rm(fn)

# temporal interpolation
u2 = [1.0 1.0]
u4 = [3.0 3.0]
knots = (range(1, stop=20, length=2),)
ϕ2 = RegisterDeformation.GridDeformation(u2, knots)
ϕ4 = RegisterDeformation.GridDeformation(u4, knots)
ϕsindex = [ϕ2, ϕ4]
ϕs = RegisterDeformation.tinterpolate(ϕsindex, [2,4], 5)
@test length(ϕs) == 5
@test eltype(ϕs) == eltype(ϕsindex)
@test all(map(x-> x == @SVector([1.0]), ϕs[1].u))
@test all(map(x-> x == @SVector([2.0]), ϕs[3].u))

# median filtering
u = rand(2, 3, 3, 9)
ϕs = RegisterDeformation.griddeformations(u, (range(1, stop=10, length=3), range(1, stop=11, length=3)))
ϕsfilt = RegisterDeformation.medfilt(ϕs, 3)
v = ϕsfilt[3].u[2,2]
v1 = median(vec(u[1,2,2,2:4]))
v2 = median(vec(u[2,2,2,2:4]))
@test v[1] == v1 && v[2] == v2

z = [(x-2.5)^2 for x = 0:5]
uold = zeros(2,6,6)
uold[1,:,:] = z .* ones(6)'
uold[2,:,:] = ones(6) .* z'
knots = (range(1, stop=20, length=6), range(1, stop=30, length=6))
ϕ = RegisterDeformation.GridDeformation(uold, knots)
ϕnew = RegisterDeformation.regrid(ϕ, (10,8))
@test ϕnew.knots[1] ≈ range(1, stop=20, length=10)
@test ϕnew.knots[2] ≈ range(1, stop=30, length=8)
ϕi = interpolate(ϕ)
for (iy,y) in enumerate(range(1, stop=30, length=8))
    ys = (y-1)*5/29
    for (ix,x) in enumerate(range(1, stop=20, length=10))
        xs = (x-1)*5/19
        @test ≈(ϕnew.u[ix,iy], [(xs-2.5)^2,(ys-2.5)^2], rtol=0.2)
        @test ϕnew.u[ix,iy] ≈ ϕi.u(x,y)
    end
end

# https://github.com/HolyLab/BlockRegistrationScheduler/pull/42#discussion_r139291426
knots = (range(0, stop=1, length=3), range(0, stop=1, length=5))
u = rand(2, 3, 5)
ϕ1 = RegisterDeformation.GridDeformation(u, knots)
ϕ2 = RegisterDeformation.GridDeformation(u, knots)
@test ϕ1 == ϕ2

# Ensure there is no conflict between Images and RegisterDeformation
using RegisterDeformation, CoordinateTransformations, Images
tform = tformrotate(pi/4)
ϕ = tform2deformation(tform, (100, 100), (7, 7))
img = rand(100, 100)
warp(img, ϕ)

nothing
