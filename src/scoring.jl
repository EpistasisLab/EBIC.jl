module scoring

export score_population

include("parameters.jl")
include("biclusterseval.jl")
include("evolution.jl")

using CUDA: @cuda, synchronize, zeros, CuArray
using DataFrames: DataFrame
using Random: rand

using .evolution: ScoredPopulation, Chromo, Population
using .biclusterseval: evaluate_fitness, compress_chromes

function score_population(
        d_input_data::CuArray{Float32,2},
        population::Population
)::ScoredPopulation
    d_fitness = zeros(Int32, length(population))

    rows_number::Int32 = size(d_input_data, 1)

    d_compressed_chromes, d_chromes_ids = compress_chromes(population)

    blocks_per_chromo = ceil(Int, rows_number / BLOCK_SIZE)

    @cuda blocks=(length(population), blocks_per_chromo) threads=(1, BLOCK_SIZE) evaluate_fitness(
        d_fitness,
        d_input_data,
        rows_number,
        d_compressed_chromes,
        d_chromes_ids
    )
    synchronize()

    return [
        chromo => score_chromo(chromo, c_fitness)
        for (chromo, c_fitness) in zip(population, Array(d_fitness))
    ]
end

function score_chromo(chromo::Chromo, fitness)::Float64
    rows = fitness
    rows <= 1 && return 0
    cols = length(chromo)
    return 2.0 ^ min(rows - MIN_NO_ROWS, 0) * log2(max(rows - 1, 0)) * cols
end

end
