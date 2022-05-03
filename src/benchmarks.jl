const DEFAULT_OUT_DIR = "results/EBIC.jl"

function benchmark_dataset(dataset_path; out_dir=DEFAULT_OUT_DIR)
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
                    "Mismatch between number of inputs and groundtruths in '$root'."
                ),
            )
        end

        test_group_results = Vector()
        for (input_path, ground_truth_path) in zip(input_paths, biclusters_paths)
            @info """#############################
            Test case  : $(basename(input_path))
            Groundtruth: $(basename(ground_truth_path))
            """

            # parameters used in our paper
            result = run_ebic(
                input_path;
                max_iterations=20_000,
                overlap_threshold=0.75,
                negative_trends=true,
                approx_trends_ratio=0.85,
                ground_truth=ground_truth_path,
                best_bclrs_stats=false,
            )

            result["input_data"] = input_path
            result["ground_truth"] = ground_truth_path

            @info """Metrics:
            Prelic relevance: $(result["relevance"])
            Prelic recovery : $(result["recovery"])
            Clustering error: $(result["ce"])
            """

            push!(test_group_results, result)
        end

        open(joinpath(out_path, "$(basename(root))_res.json"), "w") do f
            JSON.print(f, test_group_results)
        end
    end
end

function benchmark_unibic(; out_dir=DEFAULT_OUT_DIR)
    return benchmark_dataset("data/unibic/"; out_dir=out_dir)
end
function benchmark_recbic_sup(; out_dir=DEFAULT_OUT_DIR)
    return benchmark_dataset("data/recbic_sup/"; out_dir=out_dir)
end
function benchmark_recbic_main(; out_dir=DEFAULT_OUT_DIR)
    return benchmark_dataset("data/recbic_maintext/"; out_dir=out_dir)
end

benchmark_all(; out_dir=DEFAULT_OUT_DIR) = begin
    benchmark_unibic(; out_dir=out_dir)
    benchmark_recbic_main(; out_dir=out_dir)
    benchmark_recbic_sup(; out_dir=out_dir)
end
