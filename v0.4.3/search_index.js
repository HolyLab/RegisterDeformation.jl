var documenterSearchIndex = {"docs":
[{"location":"api/#API-summary-1","page":"API summary","title":"API summary","text":"","category":"section"},{"location":"api/#Creating-deformations-1","page":"API summary","title":"Creating deformations","text":"","category":"section"},{"location":"api/#","page":"API summary","title":"API summary","text":"GridDeformation\ntform2deformation\ngriddeformations\nregrid\nsimilarϕ","category":"page"},{"location":"api/#RegisterDeformation.GridDeformation","page":"API summary","title":"RegisterDeformation.GridDeformation","text":"ϕ = GridDeformation(u::Array{<:SVector}, axs) creates a deformation ϕ for an array with axes axs.  u specifies the \"pixel-wise\" displacement at a series of nodes that are evenly-spaced over the domain specified by axs (i.e., using node-vectors range(first(axs[d]), stop=last(axs[d]), length=size(u,d))). In particular, each corner of the array is the site of one node.\n\nϕ = GridDeformation(u::Array{<:SVector}, nodes) specifies the node-vectors manually. u must have dimensions equal to (length(nodes[1]), length(nodes[2]), ...).\n\nϕ = GridDeformation(u::Array{T<:Real}, ...) constructs the deformation from a \"plain\" u array. For a deformation in N dimensions, u must have N+1 dimensions, where the first dimension corresponds to the displacement along each axis (and therefore size(u,1) == N).\n\nFinally, ϕ = GridDeformation((u1, u2, ...), ...) allows you to construct the deformation using an N-tuple of shift-arrays, each with N dimensions.\n\nExample\n\nTo represent a two-dimensional deformation over a spatial region 1:100 × 1:200 (e.g., for an image of that size),\n\ngridsize = (3, 5)             # a coarse grid\nu = 10*randn(2, gridsize...)  # each displacement is 2-dimensional, typically ~10 pixels\nnodes = (range(1, stop=100, length=gridsize[1]), range(1, stop=200, length=gridsize[2]))\nϕ = GridDeformation(u, nodes) # this is a \"naive\" deformation (not ready for interpolation)\n\n\n\n\n\n","category":"type"},{"location":"api/#RegisterDeformation.tform2deformation","page":"API summary","title":"RegisterDeformation.tform2deformation","text":"ϕ = tform2deformation(tform, imgaxes, gridsize)\n\nConstruct a deformation ϕ from the affine transform tform suitable for warping arrays with axes imgaxes.  The array of grid points defining ϕ has size specified by gridsize.  The dimensionality of tform must match that specified by arraysize and gridsize.\n\nNote it's more accurate to warp(img, tform) directly; the main use of this function is to initialize a GridDeformation for later optimization.\n\n\n\n\n\n","category":"function"},{"location":"api/#RegisterDeformation.griddeformations","page":"API summary","title":"RegisterDeformation.griddeformations","text":"ϕs = griddeformations(u, nodes) constructs a vector ϕs of seqeuential deformations.  The last dimension of the array u should correspond to time; ϕs[i] is produced from u[:, ..., i].\n\n\n\n\n\n","category":"function"},{"location":"api/#RegisterDeformation.regrid","page":"API summary","title":"RegisterDeformation.regrid","text":"ϕnew = regrid(ϕ, gridsize)\n\nReparametrize the deformation ϕ so that it has a grid size gridsize.\n\nExample\n\nϕnew = regrid(ϕ, (13,11,7))\n\nfor a 3-dimensional deformation ϕ.\n\n\n\n\n\n","category":"function"},{"location":"api/#RegisterDeformation.similarϕ","page":"API summary","title":"RegisterDeformation.similarϕ","text":"ϕ = similarϕ(ϕref, coefs)\n\nCreate a deformation with the same nodes as ϕref but using coefs for the data. This is primarily useful for testing purposes, e.g., computing gradients with ForwardDiff where the elements are ForwardDiff.Dual numbers.\n\nIf ϕref is interpolating, coefs will be used for the interpolation coefficients, not the node displacements. Typically you want to create ϕref with\n\nϕref = interpolate!(copy(ϕ0))\n\nrather than interpolate(ϕ0) (see interpolate!).\n\n\n\n\n\n","category":"function"},{"location":"api/#Conversion-to-interpolating-form-1","page":"API summary","title":"Conversion to interpolating form","text":"","category":"section"},{"location":"api/#","page":"API summary","title":"API summary","text":"interpolate(::GridDeformation)\ninterpolate!(::GridDeformation)\nextrapolate(::GridDeformation)\nextrapolate!(::GridDeformation)","category":"page"},{"location":"api/#Interpolations.interpolate-Tuple{GridDeformation}","page":"API summary","title":"Interpolations.interpolate","text":"ϕi = interpolate(ϕ, BC=Flat(OnCell()))\n\nCreate a deformation ϕi suitable for interpolation, matching the displacements of ϕ.u at the nodes. A quadratic interpolation scheme is used, with default flat boundary conditions.\n\n\n\n\n\n","category":"method"},{"location":"api/#Interpolations.interpolate!-Tuple{GridDeformation}","page":"API summary","title":"Interpolations.interpolate!","text":"ϕi = interpolate!(ϕ, BC=InPlace(OnCell()))\n\nCreate a deformation ϕi suitable for interpolation, matching the displacements of ϕ.u at the nodes. ϕ is destroyed in the process.\n\nA quadratic interpolation scheme is used, with the default being to use \"InPlace\" boundary conditions.\n\nwarning: Warning\nBecause of the difference in default boundary conditions, interpolate!(copy(ϕ)) does not yield a result identical to interpolate(ϕ). When it matters, it is recommended that you annotate such calls with # not same as interpolate(ϕ) in your code.\n\n\n\n\n\n","category":"method"},{"location":"api/#Interpolations.extrapolate-Tuple{GridDeformation}","page":"API summary","title":"Interpolations.extrapolate","text":"ϕi = extrapolate(ϕ, BC=Flat(OnCell()))\n\nCreate a deformation ϕi suitable for interpolation, matching the displacements of ϕ.u at the nodes. A quadratic interpolation scheme is used, with default flat boundary conditions and Line extrapolation.\n\nwarning: Warning\nExtrapolation beyond the supported region of ϕ can yield poor results.\n\n\n\n\n\n","category":"method"},{"location":"api/#RegisterDeformation.extrapolate!-Tuple{GridDeformation}","page":"API summary","title":"RegisterDeformation.extrapolate!","text":"ϕi = extrapolate!(ϕ, BC=InPlace(OnCell()))\n\nCreate a deformation ϕi suitable for interpolation, matching the displacements of ϕ.u at the nodes. ϕ is destroyed in the process.\n\nA quadratic interpolation scheme is used, with the default being to use \"InPlace\" boundary conditions.\n\nwarning: Warning\nBecause of the difference in default boundary conditions, extrapolate!(copy(ϕ)) does not yield a result identical to extrapolate(ϕ). When it matters, it is recommended that you annotate such calls with # not same as extrapolate(ϕ) in your code.\n\n\n\n\n\n","category":"method"},{"location":"api/#Warping-images-1","page":"API summary","title":"Warping images","text":"","category":"section"},{"location":"api/#","page":"API summary","title":"API summary","text":"WarpedArray\nwarp\nwarp!\ntranslate","category":"page"},{"location":"api/#RegisterDeformation.WarpedArray","page":"API summary","title":"RegisterDeformation.WarpedArray","text":"A WarpedArray W is an AbstractArray for which W[x] = A[ϕ(x)] for some parent array A and some deformation ϕ.  The object is created lazily, meaning that computation of the displaced values occurs only when you ask for them explicitly.\n\nCreate a WarpedArray like this:\n\nW = WarpedArray(A, ϕ)\n\nwhere\n\nThe first argument A is an AbstractExtrapolation that can be evaluated anywhere.  See the Interpolations package.\nϕ is an AbstractDeformation\n\n\n\n\n\n","category":"type"},{"location":"api/#ImageTransformations.warp","page":"API summary","title":"ImageTransformations.warp","text":"wimg = warp(img, ϕ) warps the array img according to the deformation ϕ.\n\n\n\n\n\n","category":"function"},{"location":"api/#ImageTransformations.warp!","page":"API summary","title":"ImageTransformations.warp!","text":"warp!(dest, src::WarpedArray) instantiates a WarpedArray in the output dest.\n\n\n\n\n\nwarp!(dest, img, ϕ) warps img using the deformation ϕ.  The result is stored in dest.\n\n\n\n\n\nwarp!(dest, img, tform, ϕ) warps img using a combination of the affine transformation tform followed by deformation with ϕ.  The result is stored in dest.\n\n\n\n\n\nwarp!(T, io, img, ϕs; [nworkers=1]) writes warped images to disk. io is an IO object or HDF5/JLD2 dataset (the latter must be pre-allocated using d_create to be of the proper size). img is an image sequence, and ϕs is a vector of deformations, one per image in img.  If nworkers is greater than one, it will spawn additional processes to perform the deformation.\n\nAn alternative syntax is warp!(T, io, img, uarray; [nworkers=1]), where uarray is an array of u values with size(uarray)[end] == nimages(img).\n\n\n\n\n\n","category":"function"},{"location":"api/#RegisterDeformation.translate","page":"API summary","title":"RegisterDeformation.translate","text":"Atrans = translate(A, displacement) shifts A by an amount specified by displacement.  Specifically, in simple cases Atrans[i, j, ...] = A[i+displacement[1], j+displacement[2], ...].  More generally, displacement is applied only to the spatial coordinates of A.\n\nNaN is filled in for any missing pixels.\n\n\n\n\n\n","category":"function"},{"location":"api/#Composing-deformations-1","page":"API summary","title":"Composing deformations","text":"","category":"section"},{"location":"api/#","page":"API summary","title":"API summary","text":"By computing the composition ϕc = ϕ1(ϕ2), you mimic the effect of warping the image with ϕ1 and then warping the result with ϕ2.","category":"page"},{"location":"api/#","page":"API summary","title":"API summary","text":"compose","category":"page"},{"location":"api/#CoordinateTransformations.compose","page":"API summary","title":"CoordinateTransformations.compose","text":"ϕ_c = ϕ_old(ϕ_new) computes the composition of two deformations, yielding a deformation for which ϕ_c(x) ≈ ϕ_old(ϕ_new(x)). ϕ_old must be interpolating (see interpolate(ϕ_old)).\n\nϕ_c, g = compose(ϕ_old, ϕ_new) also yields the gradient g of ϕ_c with respect to u_new.  g[i,j,...] is the Jacobian matrix at grid position (i,j,...).\n\nYou can use _, g = compose(identity, ϕ_new) if you need the gradient for when ϕ_old is equal to the identity transformation.\n\n\n\n\n\nϕsi_old and ϕs_new will generate ϕs_c vector and g vector ϕsi_old is interpolated `ϕs_old: e.g) ϕsi_old = map(Interpolations.interpolate!, copy(ϕs_old))\n\n\n\n\n\n","category":"function"},{"location":"api/#Temporal-manipulations-1","page":"API summary","title":"Temporal manipulations","text":"","category":"section"},{"location":"api/#","page":"API summary","title":"API summary","text":"medfilt\ntinterpolate","category":"page"},{"location":"api/#RegisterDeformation.medfilt","page":"API summary","title":"RegisterDeformation.medfilt","text":"ϕs′ = medfilt(ϕs)\n\nPerform temporal median-filtering on a sequence of deformations. This is a form of smoothing that does not \"round the corners\" on sudden (but persistent) shifts.\n\n\n\n\n\n","category":"function"},{"location":"api/#RegisterDeformation.tinterpolate","page":"API summary","title":"RegisterDeformation.tinterpolate","text":"ϕs = tinterpolate(ϕsindex, tindex, nstack) uses linear interpolation/extrapolation in time to \"fill out\" to times 1:nstack a deformation defined intermediate times tindex . Note that ϕs[tindex] == ϕsindex.\n\n\n\n\n\n","category":"function"},{"location":"api/#Visualizing-deformations-1","page":"API summary","title":"Visualizing deformations","text":"","category":"section"},{"location":"api/#","page":"API summary","title":"API summary","text":"nodegrid\nwarpgrid","category":"page"},{"location":"api/#RegisterDeformation.nodegrid","page":"API summary","title":"RegisterDeformation.nodegrid","text":"img = nodegrid(T, nodes)\nimg = nodegrid(nodes)\nimg = nodegrid([T], ϕ)\n\nReturns an image img which has value 1 at points that are \"on the nodes.\"  If nodes/ϕ is two-dimensional, the image will consist of grid lines; for three-dimensional inputs, the image will have grid planes.\n\nThe meaning of \"on the nodes\" is that x[d] == nodes[d][i] for some dimension d and node index i. An exception is made for the edge values, which are brought inward by one pixel to prevent complete loss under subpixel deformations.\n\nOptionally specify the element type T of img.\n\nSee also warpgrid.\n\n\n\n\n\n","category":"function"},{"location":"api/#RegisterDeformation.warpgrid","page":"API summary","title":"RegisterDeformation.warpgrid","text":"img = warpgrid(ϕ; [scale=1, showidentity=false])\n\nReturns an image img that permits visualization of the deformation ϕ.  The output is a warped rectangular grid with nodes centered on the control points as specified by the nodes of ϕ.  If ϕ is a two-dimensional deformation, the image will consist of grid lines; for a three-dimensional deformation, the image will have grid planes.\n\nscale multiplies ϕ.u, effectively making the deformation stronger (for scale > 1).  This can be useful if you are trying to visualize subtle changes. If showidentity is true, an RGB image is returned that has the warped grid in magenta and the original grid in green.\n\nSee also nodegrid.\n\n\n\n\n\n","category":"function"},{"location":"api/#Low-level-utilities-1","page":"API summary","title":"Low-level utilities","text":"","category":"section"},{"location":"api/#","page":"API summary","title":"API summary","text":"eachnode\ncenteraxes","category":"page"},{"location":"api/#RegisterDeformation.eachnode","page":"API summary","title":"RegisterDeformation.eachnode","text":"iter = eachnode(ϕ)\n\nCreate an iterator for visiting all the nodes of ϕ.\n\n\n\n\n\n","category":"function"},{"location":"api/#RegisterDeformation.centeraxes","page":"API summary","title":"RegisterDeformation.centeraxes","text":"caxs = centeraxes(axs)\n\nReturn a set of axes centered on zero. Specifically, if axs[i] is a range, then caxs[i] is a UnitRange of the same length that is approximately symmetric around 0.\n\n\n\n\n\n","category":"function"},{"location":"#RegisterDeformation-1","page":"RegisterDeformation","title":"RegisterDeformation","text":"","category":"section"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"This package represents spatial deformations, sometimes also called diffeomorphisms or warps. Under a deformation, a rectangular grid might change into something like this:","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"(Image: diffeo)","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"A deformation (or warp) of space is represented by a function ϕ(x). For an image, the warped version of the image is specified by \"looking up\" the pixel value at a location ϕ(x) = x + u(x).  u(x) thus expresses the displacement, in pixels, at position x.  Note that a constant deformation, u(x) = x0, corresponds to a shift of the coordinates by x0, and therefore a shift of the image in the opposite direction.","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"Currently this package supports just one kind of deformation, one defined on a rectangular grid of u values. The package is designed for the case where the number of positions in the u grid is much smaller than the number of pixels in your image. Between grid points, the deformation can be defined by interpolation. There are two \"flavors\" of such deformations, \"naive\" (constructed directly from a u array) and \"interpolating\" (one that has been prepared for interpolation). You can prepare a \"naive\" deformation for interpolation with ϕi = interpolate(ϕ); be aware that ϕi.u ≠ ϕ.u even though , but the two","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"You can obtain a summary of the major functions in this package with ?RegisterDeformation.","category":"page"},{"location":"#Demo-1","page":"RegisterDeformation","title":"Demo","text":"","category":"section"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"Let's load an image for testing:","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"using RegisterDeformation, TestImages\nimg = testimage(\"lighthouse\")","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"DocTestSetup = quote\n    using RegisterDeformation, TestImages\n    img = testimage(\"lighthouse\")\nend","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"Now we create a deformation over the span of the image:","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"# Create a deformation\ngridsize = (5, 5)                   # a coarse grid\nu = 20*randn(2, gridsize...)        # each displacement is 2-dimensional\n# The nodes specify the location of each value in the `u` array\n# relative to the image that we want to warp. This choice spans\n# the entire image.\nnodes = map(axes(img), gridsize) do ax, g\n    range(first(ax), stop=last(ax), length=g)\nend\nϕ = GridDeformation(u, nodes)\n\n# output\n\n5×5 GridDeformation{Float64} over a domain 1.0..512.0×1.0..768.0","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"This is a \"naive\" deformation, so we can't evaluate it at an arbitrary position:","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"julia> ϕ(3.2, 1.4)\nERROR: call `ϕi = interpolate(ϕ)` and use `ϕi` for evaluating the deformation.\nStacktrace:\n [1] error(::String) at ./error.jl:33\n[...]","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"But it works if we create the corresponding interpolating deformation:","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"julia> ϕi = interpolate(ϕ)\nInterpolating 5×5 GridDeformation{Float64} over a domain 1.0..512.0×1.0..768.0\n\njulia> ϕi(3.2, 1.4)\n2-element StaticArrays.SArray{Tuple{2},Float64,1,2} with indices SOneTo(2):\n 4.5304980552861736\n 2.913923557974086","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"Now let's use this to warp the image (note it's more efficient to use ϕi here, but warp will call interpolate for you if necessary):","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"imgw = warp(img, ϕ)\naxes(imgw)\n\n# output\n\n(Base.OneTo(512), Base.OneTo(768))","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"You might get results somewhat like these:","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"Original Warped\n(Image: lh) (Image: warped)","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"The black pixels near the edge represent locations where the deformation ϕ returned a location that lies beyond the edge of the original image. The corresponding output pixels are marked with NaN values.","category":"page"},{"location":"#","page":"RegisterDeformation","title":"RegisterDeformation","text":"DocTestSetup = nothing","category":"page"}]
}
