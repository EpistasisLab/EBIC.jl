module ebicsynthtest

include("../src/Ebic.jl")

using JSON
using .Ebic: run_ebic, MAX_BICLUSTERS_NUMBER

function main()
    MAX_BICLUSTERS_NUMBER = 3

    for (root, _, files) in walkdir("data/overlap_bic")
        root == "data/overlap_bic" && continue

        println("Starting test case: '$(basename(root))'")

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
            println("Testing '$(basename(input_path))'")

            ground_truth = JSON.parsefile(bicluster_path)

            result = run_ebic(input_path, show_progress = true)

            println(result["biclusters"])
            println(ground_truth)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
