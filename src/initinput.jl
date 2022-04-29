module initinput

export init_input

using CUDA
using CSV
using Tables
using Base.Threads: @threads, nthreads, threadid

function init_input(
    input_path::String;
    ngpu::Int = 1,
    type::Type{T} = Float32,
)::Vector{CuArray{T,2}} where {T<:AbstractFloat}
    # defaults are compliant with the most common format of biclustering inputs
    data = CSV.File(input_path, drop = [1], header = false, skipto = 2) |> Tables.matrix
    data = convert(Matrix{type}, data)

    return init_input(data, ngpu = ngpu)
end

function init_input(
    input::Matrix{T};
    ngpu::Int = 1,
)::Vector{CuArray{T,2}} where {T<:AbstractFloat}
    check_cuda_devices(ngpu)

    data = coalesce.(input, typemax(T))

    nrows = size(data, 1)
    rows_per_chunk = ceil(Int, nrows / ngpu)

    d_input_data = Vector(undef, ngpu)
    @threads for dev in devices() |> collect
        device!(dev)

        chunk_begin = 1 + (threadid() - 1) * rows_per_chunk
        chunk_end = threadid() * rows_per_chunk
        arr_chunk = @view data[chunk_begin:min(chunk_end, end), :]

        d_input_data[threadid()] = CuArray(arr_chunk)
    end

    return d_input_data
end

function check_cuda_devices(ngpu::Int)
    length(devices()) < ngpu &&
        error("Not enough GPUs available: $(length(devices())) < $(ngpu).")
    nthreads() == ngpu ||
        error("The number of GPUs and threads must be equal, use '-t' option.")
end

end
