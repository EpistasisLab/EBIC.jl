const TOURNAMENT_SIZE = 50
const RATE_CROSSOVER = 0.2
const RATE_MUTATION_SWAP = 2
const RATE_MUTATION_SUBSTITUTION = 4
const RATE_MUTATION_INSERTION = 3
const RATE_MUTATION_DELETION = 1

const INIT_MAX_CHROMO_SIZE = 5

const MIN_CHROMO_SIZE = 3
const MIN_NO_ROWS = 10

const OVERLAP_PENALTY = 1.2

const EPSILON = 0.000001

const POPULATION_SIZE = 400
const REPRODUCTION_SIZE = POPULATION_SIZE / 4
const MAX_NUMBER_OF_TABU_HITS = 300

const BLOCK_SIZE = 256

# program's default arguments
const INPUT_PATH = "data/example_input.csv"

const MAX_ITERATIONS = 2_000
const MAX_BICLUSTERS_NUMBER = 3
const OVERLAP_THRESHOLD = 0.75
const NEGATIVE_TRENDS_ENABLED = true
const APPROX_TRENDS_RATIO = 0.85f0
const GPUS_NUMBER = 1

# CUSTOM TYPES
Chromo = Vector{Int}
Population = Vector{Chromo}
ScoredPopulation = Vector{Pair{Chromo,Float64}}

@enum Mutation begin
    SWAP
    SUBSTITUTION
    INSERTION
    DELETION
    CROSSOVER
end

const ALL_MUTATIONS = Dict(
    SWAP => RATE_MUTATION_SWAP,
    SUBSTITUTION => RATE_MUTATION_SUBSTITUTION,
    INSERTION => RATE_MUTATION_INSERTION,
    DELETION => RATE_MUTATION_DELETION,
)

