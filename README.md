# RegisterDeformation

[![Build Status](https://github.com/HolyLab/RegisterDeformation.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/RegisterDeformation.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/HolyLab/RegisterDeformation.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HolyLab/RegisterDeformation.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/RegisterDeformation.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/RegisterDeformation.jl/dev)

This package implements deformations (a.k.a., diffeomorphisms) for warping space.
A deformation `ϕ(x) = x + u(x)` maps every point `x` to a displaced location,
enabling image warping and spatial registration.

## Installation

This package is registered in the [HolyLab registry](https://github.com/HolyLab/HolyLabRegistry).
Add the registry once, then install normally:

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url="https://github.com/HolyLab/HolyLabRegistry"))
Pkg.add("RegisterDeformation")
```

## Quick start

```julia
using RegisterDeformation

# Create a coarse 5×5 displacement grid over a 512×768 image domain
gridsize = (5, 5)
nodes = (range(1, 512, length=5), range(1, 768, length=5))
u = zeros(2, gridsize...)          # 2D displacements, initially zero
ϕ = GridDeformation(u, nodes)

# Prepare for evaluation at arbitrary positions
ϕi = interpolate(ϕ)
ϕi(100.0, 200.0)                   # returns the displaced position

# Warp an image
using TestImages
img = testimage("lighthouse")
imgw = warp(img, ϕ)
```

See the [documentation](https://HolyLab.github.io/RegisterDeformation.jl/stable) for a full overview.

This package was split from [BlockRegistration](https://github.com/HolyLab/BlockRegistration.jl).
