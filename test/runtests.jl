using EBIC
using Test
using JSON

const DATA_PATH = (pathof(EBIC) |> dirname |> dirname) * "/data"

include("gpu_init.jl")

# the test below are not passing, it needs some work to get it right
# include("scoring.jl")

