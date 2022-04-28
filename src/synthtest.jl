include("metrics.jl")

using .metrics: eval_metrics

const DEFAULT_OUT_DIR = "results/EBIC.jl"

WARMUP_WARNING_DISPLAYED = false

function benchmark_dataset(dataset_path; out_dir = DEFAULT_OUT_DIR)
    global WARMUP_WARNING_DISPLAYED
    if !WARMUP_WARNING_DISPLAYED
        @warn "Remember to perform a warmup run of EBIC so that everything's already compiled when benchmarking!"
        WARMUP_WARNING_DISPLAYED = true
    end

    out_path = joinpath(out_dir, splitpath(dataset_path)[end])
    isdir(out_path) || mkpath(out_path)
    @info "Results will be saved in: $(realpath(out_path))"

    for (root, _, files) in walkdir(dataset_path)
        isempty(files) && continue

        @info """#############################
        TEST GROUP: '$(basename(root))'
        ###################################"""

        input_paths = Vector()
        biclusters_paths = Vector()
        for file in files
            if occursin("hidden", file)
                push!(biclusters_paths, joinpath(root, file))
            else
                push!(input_paths, joinpath(root, file))
            end
        end

        if length(input_paths) != length(biclusters_paths)
            throw(
                ErrorException(
                    "Mismatch between number of inputs and groundtruths in '$root'.",
                ),
            )
        end

        test_group_results = Vector()
        for (input_path, ground_truth_path) in zip(input_paths, biclusters_paths)
            result = benchmark_test_case(input_path, ground_truth_path)

            @info """Metrics:
            Prelic relevance   : $(result["relevance"])
            Prelic recovery    : $(result["recovery"])
            Clustering error   : $(result["ce"])
            Last iter tabu hits: $(result["last_iter_tabu_hits"])
            """

            push!(test_group_results, result)
        end

        open(joinpath(out_path, "$(basename(root))_res.json"), "w") do f
            JSON.print(f, test_group_results)
        end
    end
end

function benchmark_test_case(input_path::String, ground_truth_path::String)
    @info """#############################
    Test case  : $(basename(input_path))
    Groundtruth: $(basename(ground_truth_path))
    """

    ground_truth::Vector = JSON.parsefile(ground_truth_path)

    # parameters used in our paper
    result = run_ebic(
        input_path,
        max_iterations = 20_000,
        max_biclusters = length(ground_truth),
        overlap_threshold = 0.75,
        negative_trends = true,
        approx_trends_ratio = 0.85f0,
        best_bclrs_stats = false,
    )

    result["input_data"] = input_path
    result["ground_truth"] = ground_truth_path

    relevance, recovery, ce =
        eval_metrics(result["biclusters"], input_path, ground_truth)

    result["relevance"] = relevance
    result["recovery"] = recovery
    result["ce"] = ce
    return result
end

benchmark_unibic(; out_dir = DEFAULT_OUT_DIR) =
    benchmark_dataset("data/unibic/", out_dir = out_dir)
benchmark_recbic_sup(; out_dir = DEFAULT_OUT_DIR) =
    benchmark_dataset("data/recbic_sup/", out_dir = out_dir)
benchmark_recbic_main(; out_dir = DEFAULT_OUT_DIR) =
    benchmark_dataset("data/recbic_maintext/", out_dir = out_dir)

benchmark_all(; out_dir = DEFAULT_OUT_DIR) = begin
    benchmark_unibic(; out_dir = out_dir)
    benchmark_recbic_main(; out_dir = out_dir)
    benchmark_recbic_sup(; out_dir = out_dir)
end
