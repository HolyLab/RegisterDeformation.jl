# API summary

## Creating deformations

```@docs
GridDeformation
tform2deformation
griddeformations
regrid
```

## Warping images

```@docs
WarpedArray
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
medfilt
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
