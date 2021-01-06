module Ebic

include("parameters.jl")
include("biclusterseval.jl")
include("evolution.jl")
include("scoring.jl")
include("algorithm.jl")

using DataFrames: DataFrame
using CUDA: CuArray
using CSV: File
using Base.Order: ReverseOrdering
using DataStructures: SortedSet

using .evolution: init_population, mutate, Population
using .scoring: score_population
using .algorithm: update_rank_list!
using .biclusterseval: get_biclusters

println("This is EBIC!")


data = File(INPUT_PATH) |> DataFrame
data = data[!, 2:end]
d_input_data = CuArray(convert(Matrix{Float32}, data))

cols_number = size(d_input_data, 2)
penalties = fill(0, cols_number)

# algorithm initialization steps
println("Initializing...")
tabu_list = Set()
top_rank_list = SortedSet(Vector(), ReverseOrdering())

old_population = init_population(cols_number, tabu_list)

old_scored_population = score_population(d_input_data, old_population)

update_rank_list!(top_rank_list, old_scored_population)

println("[Top rank list]")
for (i, (chromo, score)) in enumerate(top_rank_list)
    println(score, " <-> ", chromo)
end

for i in 1:MAX_ITERATIONS
    println("[GENERATION $(i)]")
    global top_rank_list, old_scored_population, tabu_list

    new_population = Population()

    # reset penalties
    penalties = fill(0, cols_number)

    # elitism
    for (_, chromo) in top_rank_list
        !(length(new_population) < REPRODUCTION_SIZE) && break

        push!(new_population, chromo)
        for col in chromo
            penalties[col] += 1
        end
    end

    # perform mutations to recreate new population
    new_population, tabu_hits = mutate(
        new_population, old_scored_population, tabu_list, penalties, cols_number
    )

    # check if algorthm has found new soultions
    if tabu_hits >= MAX_NUMBER_OF_TABU_HITS
        println("Reached max tabu hits: '$(tabu_hits)'")
        break
    end

    # evaluate fitness for new population
    new_scored_population = score_population(d_input_data, new_population)

    # save best chromosomes
    update_rank_list!(top_rank_list, old_scored_population)

    # proceed to the next generation
    old_scored_population = new_scored_population

    # print best biclusters
    for (score, chromo) in top_rank_list
        println(chromo, " -> ", score)
    end
end

println("EBIC finished!")
println("[BICLUSTERS]")
biclusters = get_biclusters(d_input_data, [last(p) for p in top_rank_list])
for (columns, rows) in biclusters
    println("Bicluster($columns, $rows)")
end

end
