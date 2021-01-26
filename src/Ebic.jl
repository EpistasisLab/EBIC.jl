module Ebic
export run_ebic

using DataStructures: SortedSet
using ProgressMeter: next!, finish!, Progress
using ArgParse: ArgParseSettings, @add_arg_table, parse_args

include("constants.jl")
include("biclusterseval.jl")
include("evolution.jl")
include("scoring.jl")
include("algorithm.jl")

using .evolution: init_population, mutate
using .scoring: score_population
using .algorithm: update_rank_list!, ReverseOrdering
using .biclusterseval: get_biclusters, initialize_input_on_gpus

function run_ebic(
    input_path::String;
    best_bclrs_stats = true,
    verbose = false,
    max_iterations = MAX_ITERATIONS,
    max_biclusters = MAX_BICLUSTERS_NUMBER,
    overlap_threshold = OVERLAP_THRESHOLD,
    negative_trends = NEGATIVE_TRENDS_ENABLED,
    approx_trends_ratio::Float32 = APPROX_TRENDS_RATIO,
    gpus_num = GPUS_NUMBER,
)
    data_load_time = @elapsed begin
        d_input_data = initialize_input_on_gpus(input_path, gpus_num)
    end
    @debug "Loading data to GPU took: $(data_load_time)s"

    # used to evaluate the iteration and timing of the best bclrs finding
    prev_top_bclrs = Vector()
    last_top_blrs_change = (0, 0)

    if verbose
        p_bar = Progress(max_iterations, barlen = 20)
    end

    @debug "Starting algorithm..."
    # algorithm initialization steps
    start_time = time_ns()
    tabu_list = Set()
    tabu_hits = 0
    top_rank_list = SortedSet(Vector(), ReverseOrdering())

    cols_number = size(d_input_data[1], 2)

    old_population = init_population(cols_number, tabu_list)

    old_scored_population = score_population(
        d_input_data,
        old_population,
        gpus_num = gpus_num,
    )

    update_rank_list!(top_rank_list, old_scored_population, overlap_threshold)

    i = 0
    while i < max_iterations
        i += 1
        new_population = Population()

        # reset penalties
        penalties = fill(1, cols_number)

        # elitism
        for (_, chromo) in top_rank_list
            !(length(new_population) < REPRODUCTION_SIZE) && break

            push!(new_population, chromo)
            # for col in chromo # works better without
            #     penalties[col] += 1
            # end
        end

        # perform mutations to replenish new population
        new_population, tabu_hits =
            mutate(new_population, old_scored_population, tabu_list, penalties, cols_number)

        # check if algorithm has found new solutions
        if tabu_hits >= MAX_NUMBER_OF_TABU_HITS
            verbose && finish!(p_bar)
            break
        end

        # evaluate fitness for new population
        new_scored_population = score_population(
            d_input_data,
            new_population,
            gpus_num = gpus_num,
        )

        # save best chromosomes
        update_rank_list!(top_rank_list, old_scored_population, overlap_threshold)

        # proceed to the next generation
        old_scored_population = new_scored_population

        if best_bclrs_stats
            new_top_bclrs = collect(top_rank_list)[1:max_biclusters]
            if !isempty(prev_top_bclrs)
                changed = new_top_bclrs != prev_top_bclrs
                if changed
                    last_top_blrs_change = (i, time_ns() - start_time)
                    prev_top_bclrs = new_top_bclrs
                end
            else
                prev_top_bclrs = new_top_bclrs
            end
        end

        verbose && next!(p_bar)
    end

    algorithm_time = time_ns() - start_time

    biclusters = get_biclusters(
        d_input_data,
        [last(p) for p in top_rank_list],
        gpus_num,
        negative_trends,
        approx_trends_ratio,
    )

    run_summary = Dict(
        "algorithm_time" => algorithm_time / 1e9,
        "data_load_time" => data_load_time,
        "biclusters" => biclusters[1:max_biclusters],
        "performed_iters" => i,
        "last_iter_tabu_hits" => tabu_hits,
    )

    if best_bclrs_stats
        run_summary["best_bclrs_iter"] = last_top_blrs_change[1]
        run_summary["best_bclrs_time"] = last_top_blrs_change[2] / 1e9
    end

    return run_summary
end

function real_main()
    args = ArgParseSettings("""
    EBIC is a next-generation biclustering algorithm based on artificial intelligence (AI). EBIC is probably the first algorithm capable of discovering the most challenging patterns (i.e. row-constant, column-constant, shift, scale, shift-scale and trend-preserving) in complex and noisy data with average accuracy of over 90%. It is also one of the very few parallel biclustering algorithms that use at least one graphics processing unit (GPU) and is ready for big-data challenges.
    """)

    @add_arg_table args begin
        "--input", "-i"
        help = "The path to the input file."
        arg_type = String
        default = INPUT_PATH
        dest_name = "input_path"

        "--max_iterations", "-n"
        help = "The maximum number of iterations of the algorithm."
        arg_type = Int
        default = MAX_ITERATIONS

        "--biclusters_num", "-b"
        help = "The number of biclusters that will be returned in the end."
        arg_type = Int
        default = MAX_BICLUSTERS_NUMBER
        dest_name = "max_biclusters"

        "--overlap_threshold", "-o"
        help = "The maximum similarity level of each two chromosomes held in top rank list."
        arg_type = Float64
        default = OVERLAP_THRESHOLD

        "--negative_trends", "-t"
        help = "Enable negative trends."
        action = :store_true

        "--gpus_num", "-g"
        help = "The number of gpus the algorithm should run on."
        arg_type = Int
        default = GPUS_NUMBER

        "--approx_trends", "-a"
        help = ""
        arg_type = Float32
        default = APPROX_TRENDS_RATIO
        dest_name = "approx_trends_ratio"

        "--verbose", "-v"
        help = "Turn on the progress bar."
        action = :store_true

        "--best_bclrs_stats", "-s"
        help = "Evaluate resulting biclusters finding iteration and time. Enabled, it slightly worsens overall algorithm performance."
        action = :store_true
    end
    args = parse_args(args, as_symbols=true)

    results = run_ebic(args...)

    @show results
end

Base.@ccallable function julia_main()::Cint
    try
        real_main()
    catch
        Base.invokelatest(Base.display_error, Base.catch_stack())
        return 1
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    real_main()
end

end
