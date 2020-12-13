module algorithm

export update_rank_list!, ReverseOrdering, SortedSet

include("parameters.jl")
include("evolution.jl")

using DataStructures: SortedSet
using .evolution

import Base.Order: ReverseOrdering

function update_rank_list!(top_rank_list::SortedSet, scored_population::ScoredPopulation)::Nothing
    for (new_chromo, fitness) in scored_population
        (!(length(top_rank_list) < POPULATION_SIZE) ||
         !(length(top_rank_list) < REPRODUCTION_SIZE)
        ) && break

        addition_allowed = true
        for (ranked_fitness, ranked_chromo) in top_rank_list
            is_similar = eval_chromo_similarity(new_chromo, ranked_chromo) >= OVERLAP_THRESHOLD

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
        k, _ = last(top_rank_list)
        pop!(top_rank_list, k)
    end
end

end
