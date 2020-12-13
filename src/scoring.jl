module scoring

export score, score_population

include("parameters.jl")
include("evolution.jl")

using Random: rand
using .evolution: ScoredPopulation, Chromo, Population

function score(chromo::Chromo)::Float64
    return rand()
end

function score_population(population::Population)::ScoredPopulation
    return map(chromo -> Pair(chromo, score(chromo)), population)
end

end

