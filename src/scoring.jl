module scoring

export score, score_population

include("parameters.jl")
include("evolution.jl")

using CUDA
using CUDA: atomic_add!
using DataFrames
using Random: rand
using .evolution: ScoredPopulation, Chromo, Population

@assert CUDA.functional(true)

function score_population(data::DataFrame, population::Population)::ScoredPopulation
    fitness = CUDA.zeros(Int32, POPULATION_SIZE)

    input_data = CuArray(convert(Matrix{Float32}, data))

    rows_number = size(data, 1)

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

    blocks_per_chromo = ceil(Int, rows_number / BLOCK_SIZE)

    CUDA.@cuda blocks=(POPULATION_SIZE, blocks_per_chromo) threads=(1, BLOCK_SIZE) evaluate_fitness(
        fitness,
        input_data,
        rows_number,
        d_compressed_chromes,
        d_chromes_ids
    )
    CUDA.synchronize()

    return [chromo => score_chromo(chromo, c_fitness) for (chromo, c_fitness) in zip(population, Array(fitness))]
end

function score_chromo(chromo, fitness)::Float64
    rows = fitness
    rows <= 1 && return 0
    cols = length(chromo)
    return 2.0 ^ min(rows - MIN_NO_ROWS, 0) * log2(max(rows - 1, 0)) * cols
end

function evaluate_fitness(
    fitness,
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

    prev_value = input_data[idx_y, cchromes[cids[idx_x]]]

    for i in cids[idx_x]+1:cids[idx_x+1] - 1
        next_value = input_data[idx_y, cchromes[i]]

        trend_check[threadIdx().y] += next_value - prev_value + EPSILON >= 0

        prev_value = next_value
    end

    chromo_len = cids[idx_x + 1] - cids[idx_x]
    trend_check[threadIdx().y] = trend_check[threadIdx().y] + 1 >= chromo_len

    sync_threads()

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

end
