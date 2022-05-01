module algorithm

export init_top_rank_list, update_rank_list!

using DataStructures: SortedSet
using Base.Order: ReverseOrdering

include("constants.jl")
include("evolution.jl")

using .evolution: eval_chromo_similarity

function init_top_rank_list()
    return SortedSet(Vector(), ReverseOrdering())
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
