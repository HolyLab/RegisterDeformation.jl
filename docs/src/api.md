# API summary

## Overview

```@docs
RegisterDeformation
```

## Types

```@docs
AbstractDeformation
GridDeformation
NodeIterator
TransformedArray
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
ImageAxes.getindex!
```

## Composing deformations

By computing the composition `ϕc = ϕ1(ϕ2)`, you mimic the effect of warping the
image with `ϕ1` and then warping the result with `ϕ2`.

```@docs
compose
```

## Affine transforms

```@docs
tformeye
tformtranslate
tformrotate
rotation2
rotation3
rotationparameters
transform
transform!
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
RegisterDeformation.arraysize
vecindex
vecgradient!
```
