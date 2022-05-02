module scoring

export score_population

using CUDA

include("constants.jl")
include("biclusterseval.jl")
include("evolution.jl")

using .biclusterseval: count_trends, compress_chromes

function score_population(
    d_input_data::CuArray{T,2}, population::Population
) where {T<:AbstractFloat}
    compressed_chromes, chromes_ids = compress_chromes(population)

    d_trend_counts = CUDA.zeros(Int, length(population))

    d_compressed_chromes = CuArray(compressed_chromes)
    d_chromes_ids = CuArray(chromes_ids)

    nrows = size(d_input_data, 1)
    blocks_per_chromo = ceil(Int, nrows / BLOCK_SIZE)
    blocks = (length(population), blocks_per_chromo)
    threads = (1, BLOCK_SIZE)

    @cuda blocks = blocks threads = threads count_trends(
        d_trend_counts, d_input_data, nrows, d_compressed_chromes, d_chromes_ids
    )
    synchronize()

    trend_counts = Array(d_trend_counts)

    CUDA.unsafe_free!(d_trend_counts)
    CUDA.unsafe_free!(d_compressed_chromes)
    CUDA.unsafe_free!(d_chromes_ids)

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
