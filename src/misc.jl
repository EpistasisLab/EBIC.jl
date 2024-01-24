using CSV
using Tables
using CUDA

include("metrics.jl")

using .metrics: prelic_relevance, prelic_recovery, clustering_error

"""
    The defaults are compliant with the most common format of biclustering inputs.
"""
function parse_input(input_path::String)
    return Tables.matrix(CSV.File(input_path; drop=[1], header=false, skipto=2))
end

function init_input(input_path::String, ::Type{T}=Float32) where {T<:AbstractFloat}
    data = parse_input(input_path)
    data = convert(Matrix{T}, data)
    return init_input(data)
end

function init_input(input::Matrix{T})::CuArray{T,2} where {T<:AbstractFloat}
    data = coalesce.(input, typemax(T))
    d_input_data = CUDA.CuArray(data)
    return d_input_data
end

function parse_ground_truth(ground_truth_path::String)::Vector{Dict}
    ground_truth = JSON.parsefile(ground_truth_path)
    # by standard ground truth is stored in the 0-based notation, in Julia we use 1-based
    for bclr in ground_truth
        bclr["cols"] .+= 1
        bclr["rows"] .+= 1
    end
    return ground_truth
end

function eval_metrics(biclusters::Vector, ground_truth_path::String, nrow, ncol)
    ground_truth = parse_ground_truth(ground_truth_path)
    return eval_metrics(biclusters, ground_truth, nrow, ncol)
end

function eval_metrics(biclusters::Vector, ground_truth::Vector{Dict}, nrow, ncol)
    return Dict(
        "relevance" => metrics.prelic_relevance(biclusters, ground_truth),
        "recovery" => metrics.prelic_recovery(biclusters, ground_truth),
        "ce" => metrics.clustering_error(biclusters, ground_truth, nrow, ncol),
    )
end
