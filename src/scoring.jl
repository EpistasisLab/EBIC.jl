module scoring

export score, score_population

include("parameters.jl")
include("evolution.jl")

using CUDA
using DataFrames
using Random: rand
using .evolution: ScoredPopulation, Chromo, Population

@assert CUDA.functional(true)

function score_population(data::DataFrame, population::Population)::ScoredPopulation
    input_data = CuArray(convert(Matrix{Float32}, data))
    rows_number, cols_number = size(data)

    fitness = CUDA.zeros(Int32, cols_number)

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

    blocks_number::Int = ceil(Int, rows_number / BLOCK_SIZE) * POPULATION_SIZE

    CUDA.@cuda blocks=blocks_number threads=1, BLOCK_SIZE evaluate_fitness(
        fitness,
        input_data,
        rows_number,
        d_compressed_chromes,
        d_chromes_ids
    )
    CUDA.synchronize()

    return [chromo => score for (chromo, score) in zip(population, Array(fitness))]
end

function evaluate_fitness(
    fitness,
    input_data,
    rows_number::Int,
    cchromes, # compressed chromes
    cids # compressed chromes indices
)::Nothing
    idx_x = blockIdx().x * blockDim().x + threadIdx().x # bicluster/chromo number
    idx_y = blockIdx().y * blockDim().y + threadIdx().y # row number

    idx_y > rows_number && return nothing

    @cuprintln("thread $(rows_number)")

    trend_check = @cuStaticSharedMem(Int32, BLOCK_SIZE)

    trend_check[threadIdx().y] = 0

    prev_value = input_data[idx_y, cchromes[cids[idx_x]]]

    for i in cids[idx_x]+1:cids[idx_x+1]
        next_value = input_data[idx_y, cchromes[i]]

        trend_check[threadIdx().y] += next_value - prev_value + EPSILON >= 0

        prev_value = next_value
    end

    chromo_len = cids[idx_x + 1] - cids[idx_x]
    trend_check[threadIdx().y] += trend_check[threadIdx().y] + 1 >= chromo_len

    sync_threads()

    if BLOCK_SIZE == 1024
        if threadIdx().y <= 512 && idx_x + 512 <= rows_number
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 512]
        end
        sync_threads()
    end

    if BLOCK_SIZE <= 512
        if threadIdx().y <= 256 && idx_x + 256 <= rows_number
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 256]
        end
        sync_threads()
    end

    if BLOCK_SIZE <= 256
        if threadIdx().y <= 128 && idx_x + 128 <= rows_number
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 128]
        end
        sync_threads()
    end

    if BLOCK_SIZE <= 128
        if threadIdx().y <= 64 && idx_x + 64 <= rows_number
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 64]
        end
        sync_threads()
    end

    if BLOCK_SIZE <= 64
        if threadIdx().y <= 32 && idx_x + 32 <= rows_number
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 32]
        end
        sync_threads()
    end

    if BLOCK_SIZE <= 32
        if threadIdx().y <= 16 && idx_x + 16 <= rows_number
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 16]
        end
        sync_threads()
    end

    if BLOCK_SIZE <= 16
        if threadIdx().y <= 8 && idx_x + 8 <= rows_number
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 8]
        end
        sync_threads()
    end

    if BLOCK_SIZE <= 8
        if threadIdx().y <= 4 && idx_x + 4 <= rows_number
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 4]
        end
        sync_threads()
    end

    if BLOCK_SIZE <= 4
        if threadIdx().y <= 2 && idx_x + 2 <= rows_number
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 2]
        end
        sync_threads()
    end

    if BLOCK_SIZE <= 2
        if threadIdx().y <= 1 && idx_x + 1 <= rows_number
            trend_check[threadIdx().y] += trend_check[threadIdx().y + 1]
        end
        sync_threads()
    end

    if idx_y == 0
        atomic_add!(fitness[idx_x], trend_check[0])
    end
    return nothing
end

end

