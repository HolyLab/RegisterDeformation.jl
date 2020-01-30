#import BlockRegistration
using RegisterDeformation
using CoordinateTransformations, Interpolations, ImageCore, ForwardDiff
using StaticArrays, LinearAlgebra, Distributed, Statistics
using AxisArrays: AxisArray
using OffsetArrays
using Test
using RegisterDeformation, RegisterUtilities
using JLD2, HDF5, FileIO
using ImageFiltering

@testset "Nodes" begin
    nodes = (range(1, stop=15, length=5), range(1, stop=11, length=3))
    @test arraysize(nodes) == (15,11)

    nodesall = (nodes[1].*ones(length(nodes[2]))', ones(length(nodes[1])).*nodes[2]')
    for (i, knt) in enumerate(eachnode(nodes))
        @test knt[1] == nodesall[1][i]
        @test knt[2] == nodesall[2][i]
    end
end

@testset "Deformation tests" begin
    randrange(xmin, xmax) = (xmax-xmin)*rand() + xmin

    ## Deformations
    for nodes in ((range(1, stop=15, length=5),),
                  (range(1, stop=15, length=5), range(1, stop=8, length=4)),
                  (range(1, stop=15, length=3), range(1, stop=8, length=4), range(1, stop=7, length=2)),
                  (range(1, stop=15, length=5), range(1, stop=8, length=4), range(1, stop=7, length=6)))
        sz = map(length, nodes)
        rng = map(z->convert(Int, maximum(z)), nodes)
        # Zero deformation
        u = zeros(length(nodes), sz...)
        ϕ = interpolate(GridDeformation(u, nodes))
        for i = 1:10
            x = map(z->rand(1:z), rng)
            @test @inferred(ϕ(x...)) ≈ [x...]
            if all(x->x>2, sz)
                y = map(z->rand(2:z-1)+rand()-0.5, rng)
                @test ϕ(y...) ≈ [y...]
            end
        end
        # Constant shift
        dx = randn(length(nodes))
        u = zeros(length(nodes), sz...).+dx
        ϕ = interpolate(GridDeformation(u, nodes))
        for i = 1:10
            x = map(z->rand(1:z), rng)
            @test ϕ(x...) ≈ [x...]+dx
            if all(x->x>2, sz)
                y = map(z->rand(2:z-1)+rand()-0.5, rng)
                @test ϕ(y...) ≈ [y...]+dx
            end
        end
        # "Biquadratic"
        if all(x->x>2, sz)
            N = length(nodes)
            u_is = Vector{Any}(undef, N)
            fs = Vector{Any}(undef, N)
            R = CartesianIndices(sz)
            for i = 1:N
                let cntr = Float64[rand(1:z-1)+rand() for z in rng], q = rand(N)
                    f = x -> dot(q, (x-cntr).^2)
                    ui = zeros(sz)
                    for I in R
                        ui[I] = f(Float64[nodes[d][I[d]] for d = 1:N])
                    end
                    u_is[i] = reshape(ui, 1, sz...)
                    fs[i] = f
                end
            end
            u = cat(u_is..., dims = 1)
            ϕ = interpolate!(GridDeformation(copy(u), nodes), InPlaceQ(OnCell()))
            for i = 1:10
                y = Float64[randrange((node[2]+node[1])/2, (node[end]+node[end-1])/2) for node in nodes]
                dx = Float64[fs[d](y) for d=1:N]
                try
                    @test ϕ(y...) ≈ y + Float64[fs[d](y) for d=1:N]
                catch err
                    @show y nodes
                    @show [((node[2]+node[1])/2, (node[end]+node[end-1])/2) for node in nodes]
                    @show Float64[fs[d](y) for d=1:N]
                    rethrow(err)
                end
            end
        end
    end
end

@testset "tform2deformation" begin
    s = [3.3,-2.6]
    gsize = (3,2)
    A = tformtranslate(s)
    ϕ = tform2deformation(A, map(Base.OneTo, (500,480)), gsize)
    u = reshape(reinterpret(Float64, ϕ.u), tuple(2,gsize...))
    @test all(u[1,:,:] .== s[1])
    @test all(u[2,:,:] .== s[2])
end

@testset "similarϕ" begin
    uref = reshape([1.5, -0.3, 0.8], 1, 3)
    nodes = (0:5:10,)
    ϕref = GridDeformation(uref, nodes)
    unew = [0.7, 0.4, -0.33]
    ϕnew = similarϕ(ϕref, unew)
    ϕi = interpolate!(copy(ϕnew))
    @test ϕi.u(0)  ≈ SVector(0.7)
    @test ϕi.u(5)  ≈ SVector(0.4)
    @test ϕi.u(10) ≈ SVector(-0.33)

    ucoefs = copy(ϕi.u.itp.coefs)
    ϕiref = interpolate!(copy(ϕref))
    ϕi = similarϕ(ϕiref, ucoefs)
    @test ϕi.u(0)  ≈ SVector(0.7)
    @test ϕi.u(5)  ≈ SVector(0.4)
    @test ϕi.u(10) ≈ SVector(-0.33)
end

@testset "warp" begin
    # Ensure there is no conflict between ImageTransformations and RegisterDeformation
    tform = tformrotate(pi/4)
    ramp = 0.0:0.01:1
    nhalf = length(ramp) ÷ 2
    for img in (ramp*ramp', OffsetArray(ramp*ramp', -nhalf:nhalf, -nhalf:nhalf))
        imgw1 = warp(img, tform, axes(img))
        ϕ = tform2deformation(tform, axes(img), (7, 7))
        imgw2 = warp(img, ϕ)
        @test all(x->isnan(x) || abs(x) < 0.02, imgw1 - imgw2)
    end
end

@testset "WarpedArray" begin
    znan(x) = isnan(x) ? zero(x) : x

    p = (1:100).-5

    dest = Vector{Float32}(undef, 10)

    # Simple translations in 1d
    ϕ = GridDeformation([0.0,0.0]', axes(p))
    q = WarpedArray(p, ϕ);
    getindex!(dest, q, 1:10)
    @test maximum(abs.(dest - p[1:10])) < 10*eps(maximum(abs.(dest)))

    ϕ = GridDeformation([1.0,1.0]', axes(p))
    q = WarpedArray(p, ϕ);
    getindex!(dest, q, 1:10)
    @test maximum(abs.(dest - p[2:11])) < 10*eps(maximum(abs.(dest)))

    ϕ = GridDeformation([-2.0,-2.0]', axes(p))
    q = WarpedArray(p, ϕ);
    getindex!(dest, q, 1:10)
    @test maximum(abs.(dest[3:end] - p[1:8])) < 10*eps(maximum(abs.(dest[3:end])))

    getindex!(dest, q, 3:12)
    @test maximum(abs.(dest - p[1:10])) < 10*eps(maximum(abs.(dest)))

    ϕ = GridDeformation([2.0,2.0]', axes(p))
    q = WarpedArray(p, ϕ);
    getindex!(dest, q, 1:10)
    @test maximum(abs.(dest - p[3:12])) < 10*eps(maximum(abs.(dest)))

    ϕ = GridDeformation([5.0,5.0]', axes(p))
    q = WarpedArray(p, ϕ);
    getindex!(dest, q, 1:10)
    @test maximum(abs.(dest - p[6:15])) < 10*eps(maximum(abs.(dest)))

    # SubArray (test whether we can go beyond edges)
    psub = view(collect(p), 3:20)
    ϕ = GridDeformation([0.0,0.0]', axes(p))
    q = WarpedArray(psub, ϕ);
    getindex!(dest, q, -1:8)
    @test maximum(abs.(dest[3:end] - p[3:10])) < 10*eps(maximum(abs.(dest[3:end])))
    any(isnan, dest) && @warn("Some dest are NaN, not yet sure whether this is a problem")

    # Stretches
    u = [0.0,5.0,10.0]
    ϕ = interpolate(GridDeformation(u', axes(p)), Line(OnCell()))
    q = WarpedArray(p, ϕ)
    getindex!(dest, q, 1:10)
    @test abs(dest[1] - p[1]) < sqrt(eps(1.0f0))
    getindex!(dest, q, 86:95)
    @test isnan.(dest) == [falses(5);trues(5)]  # fixme
    dest2 = getindex!(zeros(Float32, 100), q, 1:100)
    @test all(abs.(diff(dest2)[26:74] .- ((u[3]-u[1])/99+1)) .< sqrt(eps(1.0f0)))

    #2d
    p = reshape(1:120, 10, 12)
    u1 = [2.0 2.0; 2.0 2.0]
    u2 = [-1.0 -1.0; -1.0 -1.0]
    ϕ = GridDeformation((u1,u2), axes(p))
    q = WarpedArray(p, ϕ)
    dest = zeros(axes(p))
    rng = (1:size(p,1),1:size(p,2))
    getindex!(dest, q, rng...)
    @test maximum(abs.(znan.(dest[1:7,2:end] - p[3:9,1:end-1]))) < 10*eps(maximum(znan.(dest)))
end

@testset "Composition" begin
    # Test that two rotations approximately compose to another rotation
    gridsize = (49,47)
    imgaxs = centeraxes(map(Base.OneTo, (200,200)))
    ϕ1 = tform2deformation(tformrotate( pi/180), imgaxs, gridsize)
    ϕ2 = tform2deformation(tformrotate(2pi/180), imgaxs, gridsize)
    ϕi = interpolate(ϕ1)
    ϕc = ϕi(ϕi)
    uc = reshape(reinterpret(Float64, ϕc.u), (2, gridsize...))
    u2 = reshape(reinterpret(Float64, ϕ2.u), (2, gridsize...))
    @test ≈(uc, u2, rtol=0.02, norm=x->maximum(abs, x))

    ## Test gradient computation

    function compare_g(g, gj, gridsize)
        for j = 1:gridsize[2], i = 1:gridsize[1]
            indx = (LinearIndices(gridsize))[i,j]
            rng = 2*(indx-1)+1:2indx
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
    function compose_u(ϕ1, u2, f, shp, imaxs)
        ϕ2 = f(GridDeformation(reshape(u2, shp), imaxs))
        ϕ = ϕ1(ϕ2)
        ret = Vector{eltype(eltype(ϕ.u))}(undef, prod(shp))
        i = 0
        for vu in ϕ.u
            for d = 1:length(vu)
                ret[i+=1] = vu[d]
            end
        end
        ret
    end

    gridsize = (11,11)
    u1 = randn(2, gridsize...)
    u2 = randn(2, gridsize...)
    imaxs = map(Base.OneTo, (100,101))
    ϕ1 = interpolate(GridDeformation(u1, imaxs))
    for f in (interpolate, identity)
        ϕ2 = f(GridDeformation(u2, imaxs))
        ϕ, g = compose(ϕ1, ϕ2)
        u2vec = vec(u2)
        gj = ForwardDiff.jacobian(u2vec -> compose_u(ϕ1, u2vec, f, size(u2), imaxs), u2vec)
        compare_g(g, gj, gridsize)
    end

    # Test identity-composition
    _, g = compose(identity, ϕ2)
    gj = zeros(2*prod(gridsize), 2*prod(gridsize))
    for i = 1:prod(gridsize)
        rng = 2*(i-1)+1:2i
        gj[rng,rng] = Matrix{Float64}(I,2,2)
    end
    compare_g(g, gj, gridsize)
end

@testset "warpgrid" begin
    nodes = (range(1, stop=100, length=4), range(1, stop=100, length=3))
    gridsize = map(length, nodes)
    ϕ = GridDeformation(5*randn(2,gridsize...), nodes)
    kg = nodegrid(nodes)
    @test eltype(kg) == Bool
    @test size(kg) == (100, 100)
    kg2 = nodegrid(ϕ)
    @test kg2 == kg
    kg = nodegrid(Float64, nodes)
    @test eltype(kg) == Float64
    A = warpgrid(ϕ)
    B = warpgrid(ϕ, scale=1.5)
    @test A != B
    @test isa(A, Matrix{Float32})
    C = warpgrid(ϕ, showidentity=true)
    @test eltype(C) == RGB{Float32}
end

@testset "warp I/O" begin
    # Writing warped data to disk
    o = ones(Float32, 5, 5)
    A = o .* reshape(1:7, (1,1,7))
    img = AxisArray(A, :y, :x, :time)
    fn = tempname()
    # With Vector{GridDeformation}
    ϕs = tighten([GridDeformation(zeros(2,3,3), axes(o)) for i = 1:nimages(img)])
    open(fn, "w") do io
        warp!(Float32, io, img, ϕs)
    end
    warped = open(fn, "r") do io
        read!(io, Array{Float32}(undef, size(img)))
    end
    @test warped == img
    # With Array{SVector}
    uarray = reshape(reinterpret(SVector{2,Float64}, zeros(2,3,3,7)), (3,3,7))
    open(fn, "w") do io
        warp!(Float32, io, img, uarray)
    end
    warped = open(fn, "r") do io
        read!(io, Array{Float32}(undef, size(img)))
    end
    @test warped == img
    # With Array{Real}
    uarray = zeros(2,3,3,7)
    fn = tempname()
    open(fn, "w") do io
        warp!(Float32, io, img, uarray)
    end
    warped = open(fn, "r") do io
        read!(io, Array{Float32}(undef, size(img)))
    end
    @test warped == img
    # Multi-process
    open(fn, "w") do io
        warp!(Float32, io, img, uarray; nworkers=3)
    end
    warped = open(fn, "r") do io
        read!(io, Array{Float32}(undef, size(img)))
    end
    @test warped == img
end

@testset "Saving deformations" begin
    # Saving arrays-of-staticarrays efficiently
    nodes = (range(1, stop=20, length=3), range(1, stop=30, length=5))
    ϕ = GridDeformation(rand(2,map(length,nodes)...), nodes)
    fn = string(tempname(), ".jld2")
    save(fn, "ϕ", ϕ)
    ϕ2 = load(fn, "ϕ")
    @test ϕ2.u == ϕ.u
    @test ϕ2.nodes == ϕ.nodes
    if HDF5.h5_get_libversion() >= v"1.10" && get(ENV, "CI", nothing) != "true"
        str = read(`h5dump $fn`, String)
        @test something(findfirst("SIMPLE { ( 5, 3, 2 ) / ( 5, 3, 2 ) }", str), 0:-1) != 0:-1||
            something(findfirst("SIMPLE { ( 30 ) / ( 30 ) }", str), 0:-1) != 0:-1||
            something(findfirst("SIMPLE { ( 2 ) / ( 2 ) }", str), 0:-1) != 0:-1
    end
    rm(fn)
end

@testset "Temporal interpolation" begin
    u2 = [1.0 1.0]
    u4 = [3.0 3.0]
    nodes = (range(1, stop=20, length=2),)
    ϕ2 = GridDeformation(u2, nodes)
    ϕ4 = GridDeformation(u4, nodes)
    ϕsindex = [ϕ2, ϕ4]
    ϕs = tinterpolate(ϕsindex, [2,4], 5)
    @test length(ϕs) == 5
    @test eltype(ϕs) == eltype(ϕsindex)
    @test all(map(x-> x == @SVector([1.0]), ϕs[1].u))
    @test all(map(x-> x == @SVector([2.0]), ϕs[3].u))
end

@testset "Median filtering" begin
    u = rand(2, 3, 3, 9)
    ϕs = griddeformations(u, (range(1, stop=10, length=3), range(1, stop=11, length=3)))
    ϕsfilt = medfilt(ϕs, 3)
    v = ϕsfilt[3].u[2,2]
    v1 = median(vec(u[1,2,2,2:4]))
    v2 = median(vec(u[2,2,2,2:4]))
    @test v[1] == v1 && v[2] == v2
end

@testset "Regridding" begin
    z = [(x-2.5)^2 for x = 0:5]
    uold = zeros(2,6,6)
    uold[1,:,:] = z .* ones(6)'
    uold[2,:,:] = ones(6) .* z'
    nodes = (range(1, stop=20, length=6), range(1, stop=30, length=6))
    ϕ = GridDeformation(uold, nodes)
    ϕnew = regrid(ϕ, (10,8))
    @test ϕnew.nodes[1] ≈ range(1, stop=20, length=10)
    @test ϕnew.nodes[2] ≈ range(1, stop=30, length=8)
    ϕi = interpolate(ϕ)
    for (iy,y) in enumerate(range(1, stop=30, length=8))
        ys = (y-1)*5/29
        for (ix,x) in enumerate(range(1, stop=20, length=10))
            xs = (x-1)*5/19
            @test ≈(ϕnew.u[ix,iy], [(xs-2.5)^2,(ys-2.5)^2], rtol=0.2)
            @test ϕnew.u[ix,iy] ≈ ϕi.u(x,y)
        end
    end
end

@testset "Equality" begin
    # https://github.com/HolyLab/BlockRegistrationScheduler/pull/42#discussion_r139291426
    nodes = (range(0, stop=1, length=3), range(0, stop=1, length=5))
    u = rand(2, 3, 5)
    ϕ1 = GridDeformation(u, nodes)
    ϕ2 = GridDeformation(u, nodes)
    @test ϕ1 == ϕ2
end

@testset "Transformation of arrays" begin
    using RegisterDeformation: center
    # TODO: get rid of dependency on ImageFiltering for faster CI times
    padwith0(A) = parent(padarray(A, Fill(0, (2,2))))

    for IT in ((BSpline(Constant())),
                    (BSpline(Linear())),
                    (BSpline(Quadratic(Flat(OnCell())))))
        A = padwith0([1 1; 1 2])
        itp = interpolate(A, IT)
        tfm = tformtranslate([1,0])
        tA = TransformedArray(extrapolate(itp, NaN), tfm)
        @test @inferred(transform(tA))[3:4,3:4] ≈ [1 2; 0 0]
        @test tA[3,3] ≈ 1
        @test tA[3,4] ≈ 2
        tfm = tformtranslate([0,1])
        tA = TransformedArray(extrapolate(itp, NaN), tfm)
        @test transform(tA)[3:4,3:4] ≈ [1 0; 2 0]
        tfm = tformtranslate([-1,0])
        tA = TransformedArray(extrapolate(itp, NaN), tfm)
        @test transform(tA)[3:4,3:4] ≈ [0 0; 1 1]
        tfm = tformtranslate([0,-1])
        tA = TransformedArray(extrapolate(itp, NaN), tfm)
        @test transform(tA)[3:4,3:4] ≈ [0 1; 0 1]

        A = padwith0([2 1; 1 1])
        itp = interpolate(A, IT)
        tfm = tformrotate(pi/2)
        tA = TransformedArray(extrapolate(itp, NaN), tfm)
        @test transform(tA)[3:4,3:4] ≈ [1 2; 1 1]
        tfm = tformrotate(-pi/2)
        tA = TransformedArray(extrapolate(itp, NaN), tfm)
        @test transform(tA)[3:4,3:4] ≈ [1 1; 2 1]

        # Check that getindex and transform yield the same results
        A = rand(6,6)
        itp = interpolate(A, IT)
        tfm = tformrotate(pi/7)∘tformtranslate(rand(2))
        tA = TransformedArray(extrapolate(itp, NaN), tfm)
        dest = transform(tA)
        tfm_recentered = AffineMap(tfm.linear, tfm.translation + center(A) - tfm.linear*center(dest))
        tA_recentered = TransformedArray(extrapolate(itp, NaN), tfm_recentered)
        nbad = 0
        for j = 1:size(dest,2), i = 1:size(dest,1)
            val = tA_recentered[i,j]
            nbad += isfinite(dest[i,j]) != isfinite(val)
            if isfinite(val) && isfinite(dest[i,j])
                @test abs(val-dest[i,j]) < 1e-12
            end
        end
        @test nbad < 3
        dest = transform(tA, origin_src=zeros(2), origin_dest=zeros(2))
        nbad = 0
        for j = 1:size(dest,2), i = 1:size(dest,1)
            val = tA[i,j]
            nbad += isfinite(dest[i,j]) != isfinite(val)
            if isfinite(val) && isfinite(dest[i,j])
                @test abs(val-dest[i,j]) < 1e-12
            end
        end
        @test nbad < 3
    end

    A = Float64[1 2 3 4; 5 6 7 8]
    tfm = tformeye(2)
    dest = zeros(2,2)

    itp = interpolate(A, BSpline(Linear()))
    tA = TransformedArray(extrapolate(itp, NaN), tfm)
    @test transform!(dest, tA) ≈ [2 3; 6 7]
    dest3 = zeros(3,3)
    @test all(isapprox.(transform!(dest3, tA), [NaN NaN NaN; 3.5 4.5 5.5; NaN NaN NaN], nans=true))

    itp = interpolate(A, BSpline(Constant()))
    tA = TransformedArray(extrapolate(itp, NaN), tfm)
    @test transform!(dest, tA) == [2 3; 6 7]

    # Transforms with non-real numbers
    using DualNumbers
    a = tformtranslate([0,dual(1,0)])
    A = reshape(1:9, 3, 3)
    At = transform(A, a)
    @test At[:,1:2] == A[:,2:3]
end
