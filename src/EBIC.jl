module EBIC

export run_ebic

include("synthtest.jl")
export benchmark_all,
    benchmark_dataset,
    benchmark_test_case,
    benchmark_unibic,
    benchmark_recbic_main,
    benchmark_recbic_sup

using JSON
using ProgressMeter: next!, finish!, Progress

include("constants.jl")  # contains default parameters
include("biclusterseval.jl")
include("evolution.jl")
include("scoring.jl")
include("initinput.jl")

using .initinput: init_input
using .evolution:
    init_population, mutate!, init_rank_list, update_rank_list!, reproduce_best_chromes!
using .scoring: score_population
using .biclusterseval: get_biclusters

"""
    run_ebic(input; <keyword_arguments>)
"""
run_ebic(input; kwargs...) = run_ebic(; input=input, kwargs...)

"""
    run_ebic(;input, <keyword_arguments>)

# Arguments
- `input::Union{String,Matrix{AbstractFloat}}`: the path to the input file with header and index or just a matrix.
- `max_iterations::Integer=$MAX_ITERATIONS`: the maximum number of iterations to perform.
- `num_biclusters::Integer=$NUM_BICLUSTERS`: the number of biclusters to be returned at the end.
- `overlap_threshold::Float64=$OVERLAP_THRESHOLD`: the maximum similarity between two chromosomes in the rank list [0, 1].
- `negative_trends::Bool=$NEGATIVE_TRENDS_ENABLED`: enable negative trends (takes effect only in the last itaration).
- `approx_trends::Float64=$APPROX_TRENDS_RATIO`: allow trends that are monotonic in the percentage of columns [0, 1] (takes effect only in the last itaration).
- `max_tabu_hits::Integer=$MAX_TABU_HITS`: the number of tabu hits that exceed causes the algorithm termination.
- `population_size::Integer=$POPULATION_SIZE`: the number of chromosomes evaluated in each iteration.
- `reproduction_size::Real=$REPRODUCTION_SIZE`: the number of best chromosomes copied from the previous iteration (elitism).
- `best_bclrs_stats::Bool=false`: track time and iteration of finding final biclusters (slightly worsens overall algorithm performance).
- `output::Bool=false`: save biclusters to a JSON file, the file name is a concatenation of the input file name and '-res.json' suffix and is saved in the current directory.

# Examples
```
julia> run_ebic(input="data/example_input.csv")
Progress: 100%|████████████████████| Time: 0:00:29
Dict{String, Any} with 4 entries:
  "tabu_hits"      => 301
  "biclusters"     => [Dict("rows"=>[16, 17, 18, 1…
  "num_iterations" => 879
  "algorithm_time" => 29.7327
```

"""
function run_ebic(;
    input,
    max_iterations=MAX_ITERATIONS,
    num_biclusters=NUM_BICLUSTERS,
    overlap_threshold=OVERLAP_THRESHOLD,
    negative_trends=NEGATIVE_TRENDS_ENABLED,
    approx_trends_ratio=APPROX_TRENDS_RATIO,
    max_tabu_hits=MAX_TABU_HITS,
    population_size=POPULATION_SIZE,
    reproduction_size=REPRODUCTION_SIZE,
    best_bclrs_stats=false,
    output=false,
)
    d_input_data = init_input(input)

    # used to evaluate the iteration and time of the final bclrs
    prev_top_bclrs = Vector()
    last_top_bclrs_change = (0, 0)

    p_bar = Progress(max_iterations; barlen=20)

    start_time = time_ns()
    tabu_hits = 0
    rank_list = init_rank_list()

    ncol = size(d_input_data[1], 2)

    old_population, tabu_list = init_population(ncol, population_size)

    old_scored_population = score_population(d_input_data, old_population)

    update_rank_list!(rank_list, old_scored_population, overlap_threshold)

    i = 0
    while i < max_iterations
        i += 1
        new_population = Population()

        # reset penalties
        penalties = fill(1, ncol)

        # elitism
        reproduce_best_chromes!(new_population, rank_list, reproduction_size)

        # perform mutations to replenish new population
        new_population, tabu_hits = mutate!(
            new_population, old_scored_population, tabu_list, penalties, ncol
        )

        # check if algorithm has found new solutions
        if tabu_hits > max_tabu_hits
            finish!(p_bar)
            break
        end

        # evaluate fitness for new population
        new_scored_population = score_population(d_input_data, new_population)

        # save best chromosomes
        update_rank_list!(rank_list, old_scored_population, overlap_threshold)

        # proceed to the next generation
        old_scored_population = new_scored_population

        if best_bclrs_stats
            new_top_bclrs = collect(rank_list)[1:num_biclusters]
            if !isempty(prev_top_bclrs)
                changed = new_top_bclrs != prev_top_bclrs
                if changed
                    last_top_bclrs_change = (i, time_ns() - start_time)
                    prev_top_bclrs = new_top_bclrs
                end
            else
                prev_top_bclrs = new_top_bclrs
            end
        end

        next!(p_bar)
    end

    fittest_chromes = [last(p) for p in rank_list]
    biclusters = get_biclusters(
        d_input_data, fittest_chromes, negative_trends, approx_trends_ratio
    )

    algorithm_time = time_ns() - start_time

    run_summary = Dict(
        "algorithm_time" => algorithm_time / 1e9,
        "biclusters" => biclusters[1:num_biclusters],
        "num_iterations" => i,
        "tabu_hits" => tabu_hits,
    )

    if best_bclrs_stats
        run_summary["best_bclrs_iter"] = last_top_bclrs_change[1]
        run_summary["best_bclrs_time"] = last_top_bclrs_change[2] / 1e9
    end

    if output && input isa String
        output_path = "$(basename(input_path))-res.json"
        open(output_path, "w") do f
            JSON.print(f, run_summary["biclusters"])
        end
        @debug "Biclusters saved to $(output_path)"
    end

    return run_summary
end

end
