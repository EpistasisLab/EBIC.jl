module ebicsynthtest

using JSON
using CSV: File
using DataFrames: DataFrame

include("../src/Ebic.jl")
include("metrics.jl")

using .Ebic: run_ebic
using .metrics: prelic_relevance, prelic_recovery, clustering_error

const DATA_DIR = "data/unibic"
const OUTPUT_DIR = "output"

function main()
    isdir(OUTPUT_DIR) || mkdir(OUTPUT_DIR)

    for (root, _, files) in walkdir(DATA_DIR)
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

            result = run_ebic(
                input_path,
                verbose = true,
                max_iterations = 20_000,
                max_biclusters = length(ground_truth),
                overlap_threshold = 0.75,
                negative_trends = true,
                approx_trends_ratio = 0.85f0,
            )

            result["input_data"] = input_path
            result["ground_truth"] = bicluster_path

            biclusters = result["biclusters"]

            relevance = prelic_relevance(biclusters, ground_truth)
            recovery = prelic_recovery(biclusters, ground_truth)

            dataset = DataFrame(File(input_path))
            nrows = size(dataset, 1)
            ncols = size(dataset, 2) - 1 # omit column with g0, g1, ...

            ce = clustering_error(biclusters, ground_truth, nrows, ncols)

            result["relevance"] = relevance
            result["recovery"] = recovery
            result["ce"] = ce

            println("Prelic relevance: $(relevance)")
            println("Prelic recovery: $(recovery)")
            println("Clustering error: $(ce)")

            push!(test_case_results, result)
        end

        open("output/$(basename(root)).json", "w") do f
            JSON.print(f, test_case_results)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
