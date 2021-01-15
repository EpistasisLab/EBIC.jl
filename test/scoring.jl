module scoringtest

using JSON
using PrettyPrinting
using Test

include("../src/biclusterseval.jl")

using .biclusterseval: initialize_input_on_gpus, get_biclusters

const DATA_DIR = "data/unibic"

function main()
    @testset "Score population" begin
    for (root, _, files) in walkdir(DATA_DIR)
        isempty(files) && continue

        println("""
        ####################################
        Test Case: '$(basename(root))'
        ####################################""")

        @testset "$(basename(root))" begin

        input_paths = Vector()
        biclusters_paths = Vector()
        for file in files
            if occursin("hidden", file)
                push!(biclusters_paths, joinpath(root, file))
            else
                push!(input_paths, joinpath(root, file))
            end
        end

        for (input_path, bicluster_path) in zip(input_paths, biclusters_paths)
            println("Testing: '$(basename(input_path))'")
            @testset "$(basename(input_path))" begin

            ground_truth = JSON.parsefile(bicluster_path)
            answers = Dict(map(o -> o["cols"] => o["rows"], ground_truth))

            population::Vector{Vector{Int64}} = collect(keys(answers))

            for chromo in population
                chromo .+= 1
            end

            d_data = initialize_input_on_gpus(input_path, 1)

            biclusters = get_biclusters(d_data, population, 1, true, 0.95f0)

            for bicluster in biclusters
                bicluster["cols"] .-= 1
                bicluster["rows"] .-= 1

                @test length(bicluster["rows"]) == length(answers[bicluster["cols"]])
            end
            end
        end
        end
    end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end
