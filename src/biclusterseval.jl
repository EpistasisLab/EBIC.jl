module biclusterseval

export get_biclusters, evaluate_fitness, compress_chromes

include("parameters.jl")
include("evolution.jl")

using CUDA
using .evolution: Population

@assert CUDA.functional(true)

function evaluate_fitness(
    fitness,
    input_data,
    rows_number::Int32,
    cchromes, # compressed chromes
    cids # compressed chromes indices
)::Nothing
    idx_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x # bicluster/chromo number
    idx_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y # row number

    trend_check = @cuStaticSharedMem(Int32, BLOCK_SIZE)
    trend_check[threadIdx().y] = 0

    idx_y > rows_number && return nothing

    evaluate_trends(trend_check, input_data, cchromes, cids)

    if BLOCK_SIZE == 1024
        if threadIdx().y <= 512 && idx_y + 512 <= rows_number
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 512]
        end
        sync_threads()
    end

    if BLOCK_SIZE <= 512
        if threadIdx().y <= 256 && idx_y + 256 <= rows_number
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 256]
        end
        sync_threads()
    end

    if BLOCK_SIZE <= 256
        if threadIdx().y <= 128 && idx_y + 128 <= rows_number
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 128]
        end
        sync_threads()
    end

    if BLOCK_SIZE <= 128
        if threadIdx().y <= 64 && idx_y + 64 <= rows_number
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 64]
        end
        sync_threads()
    end

    if BLOCK_SIZE <= 64
        if threadIdx().y <= 32 && idx_y + 32 <= rows_number
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 32]
        end
        sync_threads()
    end

    trend_check[threadIdx().y] += trend_check[threadIdx().y + 16]
    sync_threads()

    trend_check[threadIdx().y] += trend_check[threadIdx().y + 8]
    sync_threads()

    trend_check[threadIdx().y] += trend_check[threadIdx().y + 4]
    sync_threads()

    trend_check[threadIdx().y] += trend_check[threadIdx().y + 2]
    sync_threads()

    trend_check[threadIdx().y] += trend_check[threadIdx().y + 1]
    sync_threads()

    if threadIdx().y == 1
        @atomic fitness[idx_x] += trend_check[1]
    end

    return nothing
end

function get_biclusters(
        d_input_data::CuArray{Float32,2},
        population::Population
)
    rows_number = size(d_input_data, 1)

    d_rows_matrix = CUDA.zeros(Int32, (rows_number, length(population)))

    d_compressed_chromes, d_chromes_ids = compress_chromes(population)

    blocks_per_chromo = ceil(Int, rows_number / BLOCK_SIZE)

    CUDA.@cuda blocks=(length(population), blocks_per_chromo) threads=(1, BLOCK_SIZE) get_biclusters_rows(
        d_rows_matrix,
        d_input_data,
        rows_number,
        d_compressed_chromes,
        d_chromes_ids
    )
    CUDA.synchronize()

    rows_matrix = Array(d_rows_matrix)
    return [chromo => findall(i -> i != 0, rows_matrix[:, i]) for (i, chromo) in enumerate(population)]
end

function get_biclusters_rows(
    rows_matrix,
    input_data,
    rows_number::Int,
    cchromes, # compressed chromes
    cids # compressed chromes indices
)::Nothing
    idx_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x # bicluster/chromo number
    idx_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y # row number

    trend_check = @cuStaticSharedMem(Int32, BLOCK_SIZE)
    trend_check[threadIdx().y] = 0

    idx_y > rows_number && return nothing

    evaluate_trends(trend_check, input_data, cchromes, cids)

    rows_matrix[idx_y, idx_x] = trend_check[threadIdx().y]

    return nothing
end

function evaluate_trends(
    trend_check,
    input_data,
    cchromes,
    cids
)::Nothing
    idx_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x # bicluster/chromo number
    idx_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y # row number

    prev_value = input_data[idx_y, cchromes[cids[idx_x]]]

    trend_comp_count::Int32 = 0
    for i in cids[idx_x]+1:cids[idx_x+1] - 1
        next_value = input_data[idx_y, cchromes[i]]

        trend_comp_count += next_value - prev_value + EPSILON >= 0

        prev_value = next_value
    end

    chromo_len = cids[idx_x + 1] - cids[idx_x]
    trend_check[threadIdx().y] = trend_comp_count + 1 >= chromo_len

    sync_threads()
end

function compress_chromes(population::Population)
    compressed_chromes = Vector{Int32}()
    chromes_ids = Vector{Int32}()

    for chromo in population
        push!(chromes_ids, length(compressed_chromes) + 1)
        append!(compressed_chromes, chromo)
    end
    push!(chromes_ids, length(compressed_chromes) + 1)

    d_compressed_chromes = CuArray{Int32}(undef, length(compressed_chromes))
    copyto!(d_compressed_chromes, compressed_chromes)
    d_chromes_ids = CuArray{Int32}(undef, length(chromes_ids))
    copyto!(d_chromes_ids, chromes_ids)

    return d_compressed_chromes, d_chromes_ids
end

end
