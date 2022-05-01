module scoring

export score_population

using CUDA
using DataFrames: DataFrame
using Random: rand
using Base.Threads: @threads, nthreads, threadid

include("constants.jl")
include("biclusterseval.jl")
include("evolution.jl")

using .biclusterseval: count_trends, compress_chromes

function score_population(
    d_input_data::Vector{CuArray{T,2}}, population::Population
) where {T<:AbstractFloat}
    compressed_chromes, chromes_ids = compress_chromes(population)

    partial_trend_counts = Vector{Vector{Int}}(undef, nthreads())
    devices = collect(CUDA.devices())
    @threads for d_data_subset in d_input_data
        device!(devices[threadid()])

        d_trend_counts = CUDA.zeros(Int, length(population))

        d_compressed_chromes = CuArray(compressed_chromes)
        d_chromes_ids = CuArray(chromes_ids)

        nrows = size(d_data_subset, 1)
        blocks_per_chromo = ceil(Int, nrows / BLOCK_SIZE)
        blocks = (length(population), blocks_per_chromo)
        threads = (1, BLOCK_SIZE)

        @cuda blocks = blocks threads = threads count_trends(
            d_trend_counts, d_data_subset, nrows, d_compressed_chromes, d_chromes_ids
        )

        synchronize()

        partial_trend_counts[threadid()] = Array(d_trend_counts)

        CUDA.unsafe_free!(d_trend_counts)
        CUDA.unsafe_free!(d_compressed_chromes)
        CUDA.unsafe_free!(d_chromes_ids)
    end

    trend_counts = reduce(+, partial_trend_counts)

    return [
        chromo => score_chromo(chromo, c_fitness) for
        (chromo, c_fitness) in zip(population, trend_counts)
    ]
end

function score_chromo(chromo::Chromo, trend_count)
    trend_count < 2 && return zero(Float64)
    cols = length(chromo)
    return 2.0^min(trend_count - MIN_NO_ROWS, 0) * log2(max(trend_count - 1, 0)) * cols
end

end
