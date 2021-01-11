module ebicsynthtest

using JSON
using PrettyPrinting

include("../src/Ebic.jl")

using .Ebic: run_ebic

function main()
    OUTPUT_PATH = "output"
    isdir(OUTPUT_PATH) || mkdir(OUTPUT_PATH)

    for (root, _, files) in walkdir("data/unibic")
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

            # pprint(result)
            # println()

            biclusters = result["biclusters"]
            for bclr in biclusters
                bclr["cols"] .-= 1
                bclr["rows"] .-= 1
            end

            relevance = prelic_relevance(biclusters, ground_truth)
            recovery = prelic_recovery(biclusters, ground_truth)

            result["relevance"] = relevance
            result["recovery"] = recovery

            println("Prelic relevance: $(relevance)")
            println("Prelic recovery: $(recovery)")

            push!(test_case_results, result)
        end

        open("output/$(basename(root)).json", "w") do f
            JSON.print(f, JSON.json(test_case_results))
        end
    end
end

function prelic_relevance(predicted_biclusters, reference_biclusters)
    col_score = match_score(predicted_biclusters, reference_biclusters, "cols")
    row_score = match_score(predicted_biclusters, reference_biclusters, "rows")

    return sqrt(col_score * row_score)
end

function prelic_recovery(predicted_biclusters, reference_biclusters)
    return prelic_relevance(reference_biclusters, predicted_biclusters)
end

function match_score(predicted_biclusters, reference_biclusters, attr)::Float64
    isempty(predicted_biclusters) && isempty(reference_biclusters) && return 1
    isempty(predicted_biclusters) || isempty(reference_biclusters) && return 0

    return sum([
        maximum([
            length(intersect(bp[attr], br[attr])) / length(union(bp[attr], br[attr]))
            for br in reference_biclusters
        ]) for bp in predicted_biclusters
    ]) / length(predicted_biclusters)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
