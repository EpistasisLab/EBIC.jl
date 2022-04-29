using EBIC
using Test
using JSON

const DATA_PATH = (pathof(EBIC) |> dirname |> dirname) * "/data"

# these are not meant to pass, it needs some work to get it right
include("scoring.jl")

