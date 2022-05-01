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
using ArgParse: ArgParseSettings, @add_arg_table, parse_args

include("constants.jl")  # contains default parameters
include("biclusterseval.jl")
include("evolution.jl")
include("scoring.jl")
include("initinput.jl")

using .initinput: init_input
using .evolution: init_population, mutate, init_top_rank_list, update_rank_list!
using .scoring: score_population
using .biclusterseval: get_biclusters

run_ebic(input; kwargs...) = run_ebic(; input=input, kwargs...)

function run_ebic(;
    input,
    max_iterations=MAX_ITERATIONS,
    max_biclusters=MAX_BICLUSTERS_NUMBER,
    overlap_threshold=OVERLAP_THRESHOLD,
    negative_trends=NEGATIVE_TRENDS_ENABLED,
    approx_trends_ratio=APPROX_TRENDS_RATIO,
    max_tabu_hits=MAX_TABU_HITS,
    best_bclrs_stats=false,
    ngpu=GPUS_NUMBER,
    output=false,
)
    d_input_data = init_input(input; ngpu=ngpu)

    # used to evaluate the iteration and timing of the best bclrs finding
    prev_top_bclrs = Vector()
    last_top_bclrs_change = (0, 0)

    p_bar = Progress(max_iterations; barlen=20)

    # algorithm initialization steps
    start_time = time_ns()
    tabu_list = Set()
    tabu_hits = 0
    top_rank_list = init_top_rank_list()

    cols_number = size(d_input_data[1], 2)

    old_population = init_population(cols_number, tabu_list)

    old_scored_population = score_population(d_input_data, old_population)

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
        end

        # perform mutations to replenish new population
        new_population, tabu_hits = mutate(
            new_population, old_scored_population, tabu_list, penalties, cols_number
        )

        # check if algorithm has found new solutions
        if tabu_hits > max_tabu_hits
            finish!(p_bar)
            break
        end

        # evaluate fitness for new population
        new_scored_population = score_population(d_input_data, new_population)

        # save best chromosomes
        update_rank_list!(top_rank_list, old_scored_population, overlap_threshold)

        # proceed to the next generation
        old_scored_population = new_scored_population

        if best_bclrs_stats
            new_top_bclrs = collect(top_rank_list)[1:max_biclusters]
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

    biclusters = get_biclusters(
        d_input_data,
        [last(p) for p in top_rank_list],
        ngpu,
        negative_trends,
        approx_trends_ratio,
    )

    algorithm_time = time_ns() - start_time

    run_summary = Dict(
        "algorithm_time" => algorithm_time / 1e9,
        "biclusters" => biclusters[1:max_biclusters],
        "performed_iters" => i,
        "last_iter_tabu_hits" => tabu_hits,
    )

    if best_bclrs_stats
        run_summary["best_bclrs_iter"] = last_top_bclrs_change[1]
        run_summary["best_bclrs_time"] = last_top_bclrs_change[2] / 1e9
    end

    if output
        output_path = "$(basename(input_path))-res.json"
        open(output_path, "w") do f
            JSON.print(f, run_summary["biclusters"])
        end
        @debug "Biclusters save to $(output_path)"
    end

    return run_summary
end

function main()
    args = ArgParseSettings(
        """EBIC is a next-generation biclustering algorithm based on artificial intelligence (AI).
EBIC is probably the first algorithm capable of discovering the most challenging
patterns (i.e. row-constant, column-constant, shift, scale, shift-scale and
trend-preserving) in complex and noisy data with average accuracy of over 90%. It is also
one of the very few parallel biclustering algorithms that use at least one graphics
processing unit (GPU) and is ready for big-data challenges.""",
    )

    @add_arg_table args begin
        "input"
        help = "the path to the input file with header and index or just a matrix"
        arg_type = Union{String,Matrix}
        required = true

        "--max_iterations", "-n"
        help = "the maximum number of iterations to perform"
        arg_type = Int
        default = MAX_ITERATIONS

        "--biclusters_num", "-b"
        help = "the number of biclusters to be returned at the end"
        arg_type = Int
        default = MAX_BICLUSTERS_NUMBER
        dest_name = "max_biclusters"

        "--overlap_threshold", "-x"
        help = """the maximum similarity level between two chromosomes held in
        top rank list"""
        arg_type = Float64
        default = OVERLAP_THRESHOLD

        "--negative_trends", "-r"
        help = "enable negative trends (only in the last itaration)"
        action = :store_true

        "--approx_trends", "-a"
        help = """allow trends that are monotonic in percentage of columns
        (only in the last itaration)"""
        arg_type = Float64
        default = APPROX_TRENDS_RATIO
        dest_name = "approx_trends_ratio"

        "--max_tabu", "-t"
        help = "the number of tabu hits that exceed causes the algorithm termination"
        arg_type = Int
        default = MAX_TABU_HITS
        dest_name = "max_tabu_hits"

        "--best_bclrs_stats", "-s"
        help = """evaluate additional statistics regarding the best biclusters,
        slightly worsens overall algorithm performance"""
        action = :store_true

        "--ngpu", "-g"
        help = "the number of gpus the algorithm uses (not supported yet)"
        arg_type = Int
        default = GPUS_NUMBER

        "--output", "-o"
        help = """save biclusters to a JSON file, its file name is a concatenation
        of the input file name and '-res.json' suffix and is saved in the current
        directory"""
        action = :store_true
    end
    args = parse_args(args; as_symbols=true)

    results = run_ebic(; args...)

    return JSON.print(results)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
