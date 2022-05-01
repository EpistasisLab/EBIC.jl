module evolution

export init_population, mutate, init_top_rank_list, update_rank_list!

using DataStructures: SortedSet
using Base.Order: ReverseOrdering
using Random: rand
using StatsBase: sample, Weights

include("constants.jl")
include("evolutionops.jl")
# mutation_swap
# mutation_substitution
# mutation_insertion
# mutation_deletion
# crossover

function init_population(cols_number::Int, tabu_list::Set)::Population
    population = Population()
    while length(population) < POPULATION_SIZE
        chromo_size = rand(MIN_CHROMO_SIZE:INIT_MAX_CHROMO_SIZE)
        random_chromo = sample(1:cols_number, chromo_size; replace=false)

        chromo_signature = hash(random_chromo)
        if !(chromo_signature in tabu_list)
            push!(population, random_chromo)
            push!(tabu_list, chromo_signature)
        end
    end
    return population
end

function tournament_selection(
    scored_population::ScoredPopulation, penalties::Vector{Int}
)::Chromo
    best_chromo = nothing
    best_fitness = -Inf

    for _ = 1:TOURNAMENT_SIZE
        random_chromo, fitness = rand(scored_population)

        penalty = sum(map(col -> penalties[col], random_chromo))
        penalty /= length(random_chromo)
        penalty = OVERLAP_PENALTY^penalty

        if fitness - penalty >= best_fitness
            best_chromo = random_chromo
            best_fitness = fitness
        end
    end
    return best_chromo
end

function mutate(
    population::Population,
    old_scored_population::ScoredPopulation,
    tabu_list::Set,
    penalties::Vector{Int},
    cols_number::Int,
)::Tuple{Population,Int}
    tabu_hits = 0

    while length(population) < POPULATION_SIZE
        chromo1 = tournament_selection(old_scored_population, penalties)

        mutation = if rand() < RATE_CROSSOVER
            CROSSOVER
        else
            sample(collect(keys(ALL_MUTATIONS)), Weights(collect(values(ALL_MUTATIONS))))
        end

        new_chromo = if mutation == CROSSOVER
            chromo2 = tournament_selection(old_scored_population, penalties)
            crossover(chromo1, chromo2)
        elseif mutation == SWAP
            mutation_swap(chromo1)
        elseif mutation == SUBSTITUTION
            mutation_substitution(chromo1, cols_number)
        elseif mutation == INSERTION
            mutation_insertion(chromo1, cols_number)
        elseif mutation == DELETION
            mutation_deletion(chromo1)
        else
            error("Unsupported mutation $(mutation)")
        end

        @debug "$mutation: '$chromo1' -> '$new_chromo'"

        length(new_chromo) < MIN_CHROMO_SIZE && continue

        chromo_signature = hash(new_chromo)
        if !(chromo_signature in tabu_list)
            push!(population, new_chromo)
            push!(tabu_list, chromo_signature)
            for col in new_chromo
                penalties[col] += 1
            end
        else
            tabu_hits += 1
        end
    end

    return population, tabu_hits
end

function init_top_rank_list()
    return SortedSet(Vector(), ReverseOrdering())
end

function eval_chromo_similarity(chromo1::Chromo, chromo2::Chromo)::Float64
    return length(intersect(chromo1, chromo2)) / min(length(chromo1), length(chromo2))
end

function update_rank_list!(
    top_rank_list::SortedSet,
    scored_population::ScoredPopulation,
    overlap_threshold::Float64,
)
    for (new_chromo, fitness) in scored_population
        addition_allowed = true
        for (ranked_fitness, ranked_chromo) in top_rank_list
            is_similar =
                eval_chromo_similarity(new_chromo, ranked_chromo) >= overlap_threshold

            if is_similar
                if fitness > ranked_fitness
                    delete!(top_rank_list, ranked_fitness => ranked_chromo)
                else
                    addition_allowed = false
                end
                break
            end
        end

        if addition_allowed && length(new_chromo) >= MIN_CHROMO_SIZE
            insert!(top_rank_list, fitness => new_chromo)
        end
    end

    while length(top_rank_list) > REPRODUCTION_SIZE
        pop!(top_rank_list, last(top_rank_list))
    end
end

end
