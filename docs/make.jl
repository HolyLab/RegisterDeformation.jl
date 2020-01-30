using Documenter
using RegisterDeformation

makedocs(
    sitename = "RegisterDeformation",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [RegisterDeformation],
    pages = ["index.md", "api.md"]
)

deploydocs(
    repo = "github.com/HolyLab/RegisterDeformation.jl.git"
)
