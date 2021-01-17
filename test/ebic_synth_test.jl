module ebicsynthtest

using JSON
using CSV: File
using DataFrames: DataFrame

include("../src/Ebic.jl")
include("metrics.jl")

using .Ebic: run_ebic
using .metrics: prelic_relevance, prelic_recovery, clustering_error

const OUTPUT_DIR = "results"

function test_dataset(dataset_path; dry_run = false)
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
        for (input_path, bicluster_path) in zip(input_paths, biclusters_paths)
            println("##################################")
            println("Testing: '$(basename(input_path))'")
            println("Ground truth: '$(basename(bicluster_path))'")

            ground_truth = JSON.parsefile(bicluster_path)

            for bclr in ground_truth
                bclr["cols"] .+= 1
                bclr["rows"] .+= 1
            end

            dataset = DataFrame(File(input_path))
            nrows = size(dataset, 1)
            ncols = size(dataset, 2) - 1 # omit column with g0, g1, ...

            if dry_run
                println("Seems data has been loaded properly!")
                continue
            end

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
            result["ground_truth"] = bicluster_path

            biclusters = result["biclusters"]

            result["relevance"] = prelic_relevance(biclusters, ground_truth)
            result["recovery"] = prelic_recovery(biclusters, ground_truth)
            result["ce"] = clustering_error(biclusters, ground_truth, nrows, ncols)

            println("Prelic relevance: $(result["relevance"])")
            println("Prelic recovery: $(result["recovery"])")
            println("Clustering error: $(result["ce"])")

            push!(test_case_results, result)
        end

        if !dry_run
            open(joinpath(out_path, "$(basename(root))_res.json"), "w") do f
                JSON.print(f, test_case_results)
            end
        end
    end
end

test_unibic(; dry_run = false) = test_dataset("data/unibic/", dry_run = dry_run)
test_recbic_sup(; dry_run = false) = test_dataset("data/recbic_sup/", dry_run = dry_run)
test_recbic_main(; dry_run = false) =
    test_dataset("data/recbic_maintext/", dry_run = dry_run)

function main()
    test_unibic()
    test_recbic_main()
    test_recbic_sup()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
