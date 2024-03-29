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

const BLOCK_SIZE = 256

# default arguments
const MAX_ITERATIONS = 2_000
const NUM_BICLUSTERS = 3
const MAX_TABU_HITS = 300
const OVERLAP_THRESHOLD = 0.75
const APPROX_TRENDS_RATIO = 0.85
const NEGATIVE_TRENDS_ENABLED = true

const POPULATION_SIZE = 400
const REPRODUCTION_SIZE = 100

# CUSTOM TYPES
const Chromo = Vector{Int}
const Population = Vector{Chromo}
const ScoredPopulation = Vector{Pair{Chromo,Float64}}

