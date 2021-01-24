using JSON

include("../src/Ebic.jl")
include("metrics.jl")

using .Ebic: run_ebic
using .metrics: eval_metrics

const OUTPUT_DIR = "results"

function test_dataset(dataset_path)
    out_path = joinpath(OUTPUT_DIR, splitpath(dataset_path)[end])
    println(out_path)
    isdir(out_path) || mkpath(out_path)

    for (root, _, files) in walkdir(dataset_path)
        isempty(files) && continue

        println("""
        ####################################
        Starting test case: '$(basename(root))'
        ####################################""")

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
            throw(ErrorException("Something is wrong in $root."))
        end

        test_case_results = Vector()
        for (input_path, ground_truth_path) in zip(input_paths, biclusters_paths)
            println("##################################")
            println("Testing: '$(basename(input_path))'")
            println("Ground truth: '$(basename(ground_truth_path))'")

            ground_truth = JSON.parsefile(ground_truth_path)

            result = run_ebic(
                input_path,
                verbose = true,
                max_iterations = 20_000,
                max_biclusters = length(ground_truth),
                overlap_threshold = 0.75,
                negative_trends = true,
                approx_trends_ratio = 0.85f0,
                best_bclrs_stats = false,
            )

            result["input_data"] = input_path
            result["ground_truth"] = ground_truth_path

            relevance, recovery, ce = eval_metrics(result["biclusters"], input_path, ground_truth)

            result["relevance"] = relevance
            result["recovery"] = recovery
            result["ce"] = ce

            println("Prelic relevance: $(relevance)")
            println("Prelic recovery: $(recovery)")
            println("Clustering error: $(ce)")

            push!(test_case_results, result)
        end

        open(joinpath(out_path, "$(basename(root))_res.json"), "w") do f
            JSON.print(f, test_case_results)
        end
    end
end

test_unibic() = test_dataset("data/unibic/")
test_recbic_sup() = test_dataset("data/recbic_sup/")
test_recbic_main() = test_dataset("data/recbic_maintext/")

function main()
    test_unibic()
    test_recbic_main()
    test_recbic_sup()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
