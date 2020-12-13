include("../src/evolution.jl")
include("../src/algorithm.jl")

using Test
using .evolution
using .algorithm

@testset "Evaluation of chromo similarity" begin
    @test 0.5 == eval_chromo_similarity([1,2], [1,3])
    @test 1 == eval_chromo_similarity([1, 2, 3, 4], [1, 3])
    @test 0.5 == eval_chromo_similarity([1,2], [1, 3, 8, 9, 10])
end

scored_population = [[39, 99, 93, 11, 33, 85,  6] => 0.7214671139585329,
                     [4, 28, 90, 11, 33, 29] => 0.7630057288833507,
                     [67, 11, 39, 17, 23, 33, 86, 3] => 0.5962202361632121,
                     [3, 86, 33, 23, 11] => 0.0909233245733927,
                     [94, 46, 33, 23, 11] => 0.0694951511801174,
                     [29, 85, 33, 23, 11] => 0.8959085778760105,
                     [96, 62, 54, 33, 23, 11] => 0.6641474382123627]

@testset "Initalization of top rank list" begin

    top_rank_list = SortedSet(Vector(), ReverseOrdering())
    update_rank_list!(top_rank_list, scored_population)

    @test length(top_rank_list) < length(scored_population)
end

@testset "Tournament selection" begin
    @test [29, 85, 33, 23, 11] == tournament_selection(scored_population)
end
