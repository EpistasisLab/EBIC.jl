module evolution

export init_population, mutate, eval_chromo_similarity

using Random: rand
using StatsBase: sample, Weights

include("constants.jl")

function init_population(cols_number::Int, tabu_list::Set)::Population
    population = Population()
    while length(population) < POPULATION_SIZE
        chromo_size = rand(MIN_CHROMO_SIZE:INIT_MAX_CHROMO_SIZE)
        random_chromo = sample(1:cols_number, chromo_size, replace = false)

        chromo_signature = hash(random_chromo)
        if !(chromo_signature in tabu_list)
            push!(population, random_chromo)
            push!(tabu_list, chromo_signature)
        end
    end
    return population
end

function eval_chromo_similarity(chromo1::Chromo, chromo2::Chromo)::Float64
    length(intersect(Set(chromo1), Set(chromo2))) / min(length(chromo1), length(chromo2))
end

function mutation_swap(chromo::Chromo)::Chromo
    random_idx1 = rand(1:length(chromo))
    random_idx2 = rand(1:length(chromo))

    while random_idx1 == random_idx2
        random_idx2 = rand(1:length(chromo))
    end

    chromo = copy(chromo)
    chromo[random_idx1], chromo[random_idx2] = chromo[random_idx2], chromo[random_idx1]
    return chromo
end

function mutation_substitution(chromo::Chromo, cols_number::Int)::Chromo
    random_col = rand(1:cols_number)
    while random_col in chromo
        random_col = rand(1:cols_number)
    end
    random_substitution_point = rand(1:length(chromo))
    chromo = copy(chromo)
    chromo[random_substitution_point] = random_col
    return chromo
end

function mutation_insertion(chromo::Chromo, cols_number::Int)::Chromo
    random_col = rand(1:cols_number)
    while random_col in chromo
        random_col = rand(1:cols_number)
    end
    random_insertion_point = rand(1:(length(chromo) + 1))
    return insert!(copy(chromo), random_insertion_point, random_col)
end

function mutation_deletion(chromo::Chromo)::Chromo
    deletion_point = rand(1:length(chromo))
    return deleteat!(copy(chromo), deletion_point)
end

function crossover(chromo1::Chromo, chromo2::Chromo)::Chromo
    cut1_idx = rand(1:length(chromo1))
    cut2_idx = rand(1:length(chromo2))
    new_chromo = chromo1[1:cut1_idx]
    for col_number in chromo2[cut2_idx:end]
        if !(col_number in new_chromo)
            push!(new_chromo, col_number)
        end
    end
    return new_chromo
end

function tournament_selection(
    scored_population::ScoredPopulation,
    penalties::Vector{Int},
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

end
