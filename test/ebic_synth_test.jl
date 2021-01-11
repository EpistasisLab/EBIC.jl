module ebicsynthtest

using JSON
using PrettyPrinting

include("../src/Ebic.jl")

using .Ebic: run_ebic

function main()
    for (root, _, files) in walkdir("data/overlap_bic")
        root == "data/overlap_bic" && continue

        println("""
        ##################################
        Starting test case: '$(basename(root))'
        ##################################""")

        input_paths = Vector()
        biclusters_paths = Vector()
        for file in files
            if occursin("csv", file)
                push!(input_paths, joinpath(root, file))
            else
                push!(biclusters_paths, joinpath(root, file))
            end
        end

        for (input_path, bicluster_path) in zip(input_paths, biclusters_paths)
            println("##################################")
            println("Testing: '$(basename(input_path))'")
            println("Ground truth: '$(basename(bicluster_path))'")

            ground_truth = JSON.parsefile(bicluster_path)

            result = run_ebic(input_path, verbose = true)

            pprint(result)
            println()

            biclusters = result["biclusters"]
            for bclr in biclusters
                bclr["cols"] .-= 1
                bclr["rows"] .-= 1
            end

            println("Prelic relevance: $(prelic_relevance(biclusters, ground_truth))")
            println("Prelic recovery: $(prelic_recovery(biclusters, ground_truth))")
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
