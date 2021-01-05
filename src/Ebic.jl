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
cols_number = size(df, 2)
penalties = fill(0, cols_number)

# algorithm initialization steps
println("Initializing...")
tabu_list = Set()
top_rank_list = SortedSet(Vector(), ReverseOrdering())

old_population = init_population(cols_number, tabu_list)

old_scored_population = score_population(df, old_population)
ss_population = sort(Dict(old_scored_population), byvalue=true,
                     order=ReverseOrdering())
for (i, (chromo, score)) in enumerate(ss_population)
    println(score, " <-> ", chromo)
end

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
    new_scored_population = score_population(df, new_population)

    # save best chromosomes
    update_rank_list!(top_rank_list, old_scored_population)

    # proceed to the next generation
    old_scored_population = new_scored_population

    for (i, (score, chromo)) in enumerate(top_rank_list)
        # i > 3 && break
        println(chromo, " -> ", score)
    end
end

println("EBIC finished!")
end
