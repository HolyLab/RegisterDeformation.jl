# API summary

## Types

```@docs
AbstractDeformation
GridDeformation
NodeIterator
WarpedArray
```

## Creating deformations

```@docs
tform2deformation
griddeformations
regrid
similar_deformation
```

## Conversion to interpolating form

```@docs
interpolate(::GridDeformation)
interpolate!(::GridDeformation)
extrapolate(::GridDeformation)
extrapolate!(::GridDeformation)
```

## Warping images

```@docs
warp
warp!
translate
```

## Composing deformations

By computing the composition `ϕc = ϕ1(ϕ2)`, you mimic the effect of warping the
image with `ϕ1` and then warping the result with `ϕ2`.

```@docs
compose
```

## Temporal manipulations

```@docs
tmedfilt
tmedfilt!
tinterpolate
```

## Visualizing deformations

```@docs
nodegrid
warpgrid
```

## Low-level utilities

```@docs
eachnode
centeraxes
```
