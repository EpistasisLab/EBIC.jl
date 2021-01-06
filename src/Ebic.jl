module Ebic

include("parameters.jl")
include("biclusterseval.jl")
include("evolution.jl")
include("scoring.jl")
include("algorithm.jl")

using DataFrames: DataFrame
using CUDA: CuArray
using CSV: File
using Base.Order: ReverseOrdering
using DataStructures: SortedSet
using ProgressMeter: next!, finish!, Progress, BarGlyphs

using .evolution: init_population, mutate, Population
using .scoring: score_population
using .algorithm: update_rank_list!
using .biclusterseval: get_biclusters

println("This is EBIC!")

function run_ebic(input_path::String; bclrs_iter = true, show_progress = false)
    data_load_time = @elapsed begin
        data = File(input_path) |> DataFrame
        data = data[!, 2:end]
        d_input_data = CuArray(convert(Matrix{Float32}, data))
    end
    @debug "Loading data to GPU took: $(data_load_time)s"

    # used to evaluate iteration of finding best bclrs
    prev_top_bclrs = Vector()
    last_top_blrs_change_it = 0

    if show_progress
        p_bar = Progress(MAX_ITERATIONS, barglyphs=BarGlyphs("[=> ]"), barlen=20)
    end

    println("Starting algorithm...")
    algorithm_time = @elapsed begin
    # algorithm initialization steps
    tabu_list = Set()
    tabu_hits = 0
    top_rank_list = SortedSet(Vector(), ReverseOrdering())

    cols_number = size(d_input_data, 2)

    old_population = init_population(cols_number, tabu_list)

    old_scored_population = score_population(d_input_data, old_population)

    update_rank_list!(top_rank_list, old_scored_population)

    i = 0
    while i < MAX_ITERATIONS
        i += 1
        new_population = Population()

        # reset penalties
        penalties = fill(0, cols_number)

        # elitism
        for (_, chromo) in top_rank_list
            !(length(new_population) < REPRODUCTION_SIZE) && break

            push!(new_population, chromo)
            for col in chromo
                penalties[col] += 1
            end
        end

        # perform mutations to replenish new population
        new_population, tabu_hits = mutate(
            new_population, old_scored_population, tabu_list, penalties, cols_number
        )

        # check if algorithm has found new solutions
        if tabu_hits >= MAX_NUMBER_OF_TABU_HITS
            show_progress && finish!(p_bar)
            break
        end

        # evaluate fitness for new population
        new_scored_population = score_population(d_input_data, new_population)

        # save best chromosomes
        update_rank_list!(top_rank_list, old_scored_population)

        # proceed to the next generation
        old_scored_population = new_scored_population

        if bclrs_iter
            new_top_bclrs = collect(top_rank_list)[1:MAX_BICLUSTERS_NUMBER]
            if !isempty(prev_top_bclrs)
                changed = new_top_bclrs != prev_top_bclrs
                if changed
                    last_top_blrs_change_it = i
                    prev_top_bclrs = new_top_bclrs
                end
            else
                prev_top_bclrs = new_top_bclrs
            end
        end

        show_progress && next!(p_bar)
    end
    end

    biclusters = get_biclusters(d_input_data, [last(p) for p in top_rank_list])
    biclusters_info = ""
    for (i, (columns, rows)) in enumerate(biclusters)
        i > MAX_BICLUSTERS_NUMBER && break
        biclusters_info *= "\n\tBicluster($columns, $rows)"
    end

    top_rank_list_info = ""
    for (i, (score, chromo)) in enumerate(top_rank_list)
        i > MAX_BICLUSTERS_NUMBER && break
        top_rank_list_info *= "\n\t$chromo -> $score"
    end

    @debug """[ALGORITHM RUN SUMMARY]
    Algorithm time: $(algorithm_time)s
    Input data load: $(data_load_time)s
    Top rank list: $(top_rank_list_info)
    Biclusters: $(biclusters_info)
    Tabu hits in the last iteration: $(tabu_hits)
    Iterations: $(i)
    Biclusters found in iter: $(last_top_blrs_change_it)
    """
    return Dict(
        "algorithm_time" => algorithm_time,
        "data_load_time" => data_load_time,
        "top_rank_list" => collect(top_rank_list)[1:MAX_BICLUSTERS_NUMBER],
        "biclusters" => biclusters[1:MAX_BICLUSTERS_NUMBER],
        "last_it_tabu_hits" => tabu_hits,
        "performed_iters" => i,
        "best_bclrs_iter" => last_top_blrs_change_it,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_ebic(INPUT_PATH)
end

end
