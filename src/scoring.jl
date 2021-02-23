module scoring

export score_population

using CUDA
using DataFrames: DataFrame
using Random: rand
using Base.Threads: @threads, nthreads, threadid

include("constants.jl")
include("biclusterseval.jl")
include("evolution.jl")

using .biclusterseval: evaluate_fitness, compress_chromes

function score_population(
    d_input_data::Vector{CuArray{Float32,2}},
    population::Population;
    gpus_num::Int = 1,
    return_score = true,
)
    compressed_chromes, chromes_ids = compress_chromes(population)

    partial_fitnesses = Vector(undef, nthreads())
    @threads for (dev, d_data_subset) in collect(zip(devices(), d_input_data))
        device!(dev)

        d_fitness = CUDA.zeros(Int32, length(population))

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

        synchronize()

        partial_fitnesses[threadid()] = Array(d_fitness)

        CUDA.unsafe_free!(d_fitness)
        CUDA.unsafe_free!(d_compressed_chromes)
        CUDA.unsafe_free!(d_chromes_ids)
    end

    fitness = reduce(+, partial_fitnesses)

    return if return_score
        [chromo => score_chromo(chromo, c_fitness) for
        (chromo, c_fitness) in zip(population, fitness)]
    else
        collect(zip(population, fitness))
    end
end

function score_chromo(chromo::Chromo, fitness)::Float64
    rows = fitness
    rows <= 1 && return 0
    cols = length(chromo)
    return 2.0^min(rows - MIN_NO_ROWS, 0) * log2(max(rows - 1, 0)) * cols
end

end
