using EBIC.initinput: init_input
using CUDA

example_input_path = DATA_PATH * "/example_input.csv"

@testset "Input data initialization" begin
    input = init_input(example_input_path, ngpu = 1)
    @test length(input) == 1
    @test size(input[1]) == (150, 100)
    @test typeof(input[1]) == CuArray{Float32,2,CUDA.Mem.DeviceBuffer}

    input = init_input(example_input_path, ngpu = 1, type = Float64)
    @test typeof(input[1]) == CuArray{Float64,2,CUDA.Mem.DeviceBuffer}
end

