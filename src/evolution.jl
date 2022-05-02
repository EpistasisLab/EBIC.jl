module evolution

export init_population, mutate!, init_rank_list, update_rank_list!

using DataStructures: SortedSet, SortedDict
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

@enum Mutation begin
    SUBSTITUTION
    INSERTION
    SWAP
    DELETION
    CROSSOVER
end

const ALL_MUTATIONS = SortedDict(
    SWAP => RATE_MUTATION_SWAP,
    SUBSTITUTION => RATE_MUTATION_SUBSTITUTION,
    INSERTION => RATE_MUTATION_INSERTION,
    DELETION => RATE_MUTATION_DELETION,
)

function init_population(ncol::Int, population_size, rng)
    tabu_list = Set{UInt64}()
    population = Population()
    while length(population) < population_size
        chromo_size = rand(rng, MIN_CHROMO_SIZE:INIT_MAX_CHROMO_SIZE)
        random_chromo = sample(rng, 1:ncol, chromo_size; replace=false)

        chromo_signature = hash(random_chromo)
        if !(chromo_signature in tabu_list)
            push!(population, random_chromo)
            push!(tabu_list, chromo_signature)
        end
    end

    return population, tabu_list
end

function tournament_selection(
    scored_population::ScoredPopulation, penalties::Vector{Int}, rng
)
    best_chromo, best_fitness = rand(rng, scored_population)

    i = 1
    while i <= TOURNAMENT_SIZE
        i += 1
        random_chromo, fitness = rand(rng, scored_population)

        penalty::Float64 = sum(penalties[random_chromo])
        penalty /= length(random_chromo)
        penalty = OVERLAP_PENALTY^penalty

        if fitness - penalty >= best_fitness
            best_chromo = random_chromo
            best_fitness = fitness
        end
    end
    return best_chromo
end

function mutate!(
    population::Population,
    population_size,
    old_scored_population::ScoredPopulation,
    tabu_list::Set,
    penalties::Vector{Int},
    ncol::Int,
    rng,
)
    tabu_hits = zero(Int)

    while length(population) < population_size
        chromo1 = tournament_selection(old_scored_population, penalties, rng)

        mutation = if rand(rng) < RATE_CROSSOVER
            CROSSOVER
        else
            mutations = collect(keys(ALL_MUTATIONS))
            weights = Weights(collect(values(ALL_MUTATIONS)))
            sample(rng, mutations, weights)
        end

        new_chromo = if mutation == CROSSOVER
            chromo2 = tournament_selection(old_scored_population, penalties, rng)
            crossover(chromo1, chromo2, rng)
        elseif mutation == SWAP
            mutation_swap(chromo1, rng)
        elseif mutation == SUBSTITUTION
            mutation_substitution(chromo1, ncol, rng)
        elseif mutation == INSERTION
            mutation_insertion(chromo1, ncol, rng)
        else
            mutation == DELETION
            mutation_deletion(chromo1, rng)
        end

        length(new_chromo) < MIN_CHROMO_SIZE && continue

        chromo_signature = hash(new_chromo)
        if !(chromo_signature in tabu_list)
            push!(population, new_chromo)
            push!(tabu_list, chromo_signature)
            penalties[new_chromo] .+= 1
        else
            tabu_hits += 1
        end
    end

    return population, tabu_hits
end

function init_rank_list()
    return SortedSet{Pair{Float64,Chromo}}([], ReverseOrdering())
end

function eval_chromo_similarity(chromo1::Chromo, chromo2::Chromo)::Float64
    return length(intersect(chromo1, chromo2)) / min(length(chromo1), length(chromo2))
end

function update_rank_list!(
    top_rank_list::SortedSet,
    scored_population::ScoredPopulation,
    overlap_threshold::Float64,
    max_rank_list_size::Real,
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

    while length(top_rank_list) > max_rank_list_size
        pop!(top_rank_list, last(top_rank_list))
    end

    return nothing
end

end
