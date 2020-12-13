module Ebic

include("parameters.jl")
include("evolution.jl")
include("scoring.jl")
include("algorithm.jl")

using DataFrames
using CSV

using .evolution
using .scoring
using .algorithm

println("This is EBIC!")


df = CSV.File(INPUT_PATH, delim = '\t') |> DataFrame
df = df[!, 2:end]
cols_number = size(df)[2]

# algorithm initialization steps
tabu_list = Set()
top_rank_list = SortedSet(Vector(), ReverseOrdering())

old_population = init_population(cols_number, tabu_list)

old_scored_population = score_population(old_population)

update_rank_list!(top_rank_list, old_scored_population)

for i in 1:MAX_ITERATIONS
    println("[GENERATION $(i)]")
    global top_rank_list, old_scored_population, tabu_list

    new_population = Population()

    # elitism
    for (_, chromo) in top_rank_list
        !(length(new_population) < REPRODUCTION_SIZE) && break
        push!(new_population, chromo)
    end

    # perform mutations to recreate new population
    new_population, tabu_hits = mutate(new_population, old_scored_population, tabu_list, cols_number)

    # check if algorthm has found new soultions
    if tabu_hits >= MAX_NUMBER_OF_TABU_HITS
        println("Reached max tabu hits: '$(tabu_hits)'")
        break
    end

    # evaluate fitness for new population
    new_scored_population = score_population(new_population)

    # save best chromosomes
    update_rank_list!(top_rank_list, old_scored_population)

    # move to next generation
    old_scored_population = new_scored_population
end

println("EBIC finished!")
end
