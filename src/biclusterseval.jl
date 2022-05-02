module biclusterseval

export get_biclusters, count_trends, compress_chromes

using CUDA
using Base.Threads: @threads, nthreads, threadid

include("constants.jl")

function count_trends(
    trend_counts,
    input_data,
    nrows,
    cchromes, # compressed chromes
    cids, # compressed chromes indices
)::Nothing
    idx_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x # bicluster/chromo number
    idx_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y # row number

    trend_check = @cuStaticSharedMem(Int32, BLOCK_SIZE)
    trend_check[threadIdx().y] = 0

    idx_y > nrows && return nothing

    evaluate_trends(trend_check, input_data, cchromes, cids)

    if BLOCK_SIZE == 1024
        if threadIdx().y <= 512 && idx_y + 512 <= nrows
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 512]
        end
        sync_threads()
    end

    if BLOCK_SIZE >= 512
        if threadIdx().y <= 256 && idx_y + 256 <= nrows
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 256]
        end
        sync_threads()
    end

    if BLOCK_SIZE >= 256
        if threadIdx().y <= 128 && idx_y + 128 <= nrows
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 128]
        end
        sync_threads()
    end

    if BLOCK_SIZE >= 128
        if threadIdx().y <= 64 && idx_y + 64 <= nrows
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 64]
        end
        sync_threads()
    end

    if BLOCK_SIZE >= 64
        if threadIdx().y <= 32 && idx_y + 32 <= nrows
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 32]
        end
        sync_threads()
    end

    if threadIdx().y <= 32
        trend_check[threadIdx().y] += trend_check[threadIdx().y + 16]
        sync_warp()
        trend_check[threadIdx().y] += trend_check[threadIdx().y + 8]
        sync_warp()
        trend_check[threadIdx().y] += trend_check[threadIdx().y + 4]
        sync_warp()
        trend_check[threadIdx().y] += trend_check[threadIdx().y + 2]
        sync_warp()
        trend_check[threadIdx().y] += trend_check[threadIdx().y + 1]

        if threadIdx().y == 1
            CUDA.@atomic trend_counts[idx_x] += trend_check[1]
        end
    end

    return nothing
end

function get_biclusters(
    d_input_data::Vector{CuArray{T,2}},
    population::Population,
    negative_trends,
    approx_trends_ratio,
) where {T<:AbstractFloat}
    compressed_chromes, chromes_ids = compress_chromes(population)

    matrices = Vector(undef, nthreads())
    devices = collect(CUDA.devices())
    @threads for d_data_subset in d_input_data
        device!(devices[threadid()])

        d_compressed_chromes = CuArray(compressed_chromes)
        d_chromes_ids = CuArray(chromes_ids)

        nrows = size(d_data_subset, 1)
        blocks_per_chromo = ceil(Int, nrows / BLOCK_SIZE)
        blocks = (length(population), blocks_per_chromo)
        threads = (1, BLOCK_SIZE)

        d_matrix = CUDA.zeros(Int, (nrows, length(population)))

        @cuda blocks = blocks threads = threads get_biclusters_rows(
            d_matrix,
            d_data_subset,
            nrows,
            d_compressed_chromes,
            d_chromes_ids,
            negative_trends,
            approx_trends_ratio,
        )
        synchronize()

        matrices[threadid()] = Array(d_matrix)

        CUDA.unsafe_free!(d_matrix)
        CUDA.unsafe_free!(d_data_subset)
        CUDA.unsafe_free!(d_compressed_chromes)
        CUDA.unsafe_free!(d_chromes_ids)
    end

    matrix = vcat(matrices...)

    return [
        Dict("cols" => chromo, "rows" => findall(isone, matrix[:, i])) for
        (i, chromo) in enumerate(population)
    ]
end

function get_biclusters_rows(
    rows_matrix,
    input_data,
    nrows,
    cchromes, # compressed chromes
    cids, # compressed chrome indices
    negative_trends,
    approx_trends_ratio,
)::Nothing
    idx_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x # bicluster/chromo number
    idx_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y # row number

    trend_check = @cuStaticSharedMem(Int32, BLOCK_SIZE)
    trend_check[threadIdx().y] = 0

    idx_y > nrows && return nothing

    evaluate_trends(
        trend_check,
        input_data,
        cchromes,
        cids,
        approx_trends_ratio=approx_trends_ratio,
        trend_sign=1,
    )

    if negative_trends
        evaluate_trends(
            trend_check,
            input_data,
            cchromes,
            cids,
            approx_trends_ratio=approx_trends_ratio,
            trend_sign=-1,
        )
    end

    rows_matrix[idx_y, idx_x] = trend_check[threadIdx().y]

    return nothing
end

function evaluate_trends(
    trend_check,
    input_data::CUDA.CuDeviceMatrix{T,1},
    cchromes,
    cids;
    approx_trends_ratio=1,
    trend_sign=1,
) where {T<:AbstractFloat}
    idx_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x # bicluster/chromo number
    idx_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y # row number

    prev_value = input_data[idx_y, cchromes[cids[idx_x]]]

    trend_count = 0
    for i = (cids[idx_x] + 1):(cids[idx_x + 1] - 1)
        next_value = input_data[idx_y, cchromes[i]]

        trend_count +=
            trend_sign * (next_value - prev_value + EPSILON) >= 0 &&
            prev_value != typemax(T)

        prev_value = next_value
    end

    chromo_len = cids[idx_x + 1] - cids[idx_x]
    trend_check[threadIdx().y] += trend_count + 1 >= chromo_len * approx_trends_ratio

    sync_threads()

    return nothing
end

function compress_chromes(population::Population)
    compressed_chromes = Vector{Int32}()
    chromes_ids = Vector{Int32}()

    for chromo in population
        push!(chromes_ids, length(compressed_chromes) + 1)
        append!(compressed_chromes, chromo)
    end
    push!(chromes_ids, length(compressed_chromes) + 1)

    return compressed_chromes, chromes_ids
end

end
