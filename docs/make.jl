using Documenter
using RegisterDeformation
using ImageAxes

makedocs(
    sitename = "RegisterDeformation",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [RegisterDeformation],
    checkdocs = :exports,
    pages = ["index.md", "api.md"]
)

deploydocs(
    repo = "github.com/HolyLab/RegisterDeformation.jl.git",
    devbranch = "master",
)
