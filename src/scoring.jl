module scoring

export score_population

using CUDA
using DataFrames: DataFrame
using Random: rand
using Base.Threads: @threads

include("constants.jl")
include("biclusterseval.jl")
include("evolution.jl")

using .biclusterseval: evaluate_fitness, compress_chromes

function score_population(
    d_input_data::Vector{CuArray{Float32,2}},
    population::Population,
    gpus_num::Int,
)::ScoredPopulation
    d_fitnesses = [CUDA.zeros(Int32, length(population)) for _ = 1:gpus_num]
    compressed_chromes, chromes_ids = compress_chromes(population)

    @threads for (dev, d_data_subset, d_fitness) in
                 collect(zip(devices(), d_input_data, d_fitnesses))
        device!(dev)

        rows_number::UInt32 = size(d_data_subset, 1)

        blocks_per_chromo = ceil(Int, rows_number / BLOCK_SIZE)

        d_compressed_chromes = CuArray{Int32}(undef, length(compressed_chromes))
        copyto!(d_compressed_chromes, compressed_chromes)
        d_chromes_ids = CuArray{Int32}(undef, length(chromes_ids))
        copyto!(d_chromes_ids, chromes_ids)

        @cuda blocks = (length(population), blocks_per_chromo) threads = (1, BLOCK_SIZE) evaluate_fitness(
            d_fitness,
            d_data_subset,
            rows_number,
            d_compressed_chromes,
            d_chromes_ids,
        )
    end

    synchronize()

    fitness = Base.zeros(Int32, length(population))

    for d_fitness in d_fitnesses
        fitness .+= Array(d_fitness)
    end

    return [chromo => score_chromo(chromo, c_fitness) for
    (chromo, c_fitness) in zip(population, fitness)]
end

function score_chromo(chromo::Chromo, fitness)::Float64
    rows = fitness
    rows <= 1 && return 0
    cols = length(chromo)
    return 2.0^min(rows - MIN_NO_ROWS, 0) * log2(max(rows - 1, 0)) * cols
end

end
