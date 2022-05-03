# EBIC.jl

EBIC is a next-generation biclustering algorithm based on artificial intelligence. EBIC is probably the first algorithm capable of discovering the most challenging patterns (i.e. row-constant, column-constant, shift, scale, shift-scale and trend-preserving) in complex and noisy data with average accuracy of over 90%. It is also one of the very few parallel biclustering algorithms that use at least one graphics processing unit (GPU) and is ready for big-data challenges.

The repository contains the new version of [EBIC](https://github.com/EpistasisLab/ebic) rewritten and improved.

<p align="center">
    <img src="figures/ce_by_alg_unibic.png" width="70%">
</p>

## Requirements

- Julia 1.6 or higher
- CUDA-capable GPU with compute capability 5.0 (Maxwell) or higher

## Setting up project

1. Clone the project.

3. Start Julia in the repository root.

```bash
julia --project
```

4. Enter the pkg mode (hit `]`) and install all dependencies.

```julia
(EBIC) pkg> instantiate
```

5. Start a quick test on `data/example_input.csv` to make sure everything works.

```julia
julia> using EBIC
julia> run_ebic("data/example_input.csv")
Progress: 100%|████████████████████| Time: 0:00:23 (11.68 ms/it)
Dict{String, Any} with 4 entries:
  "biclusters"     => [Dict("rows"=>[31, 32, 33, …
  "num_iterations" => 732
  "algorithm_time" => 33.9547
```

or provide ground truth to get biclustering metrics right away:

```julia
julia> using EBIC
julia> run_ebic("data/unibic/narrow_bic/narrow_100_10/narrow_100_10_data1.txt",
    "data/unibic/narrow_bic/narrow_100_10/narrow_100_10_data1_hiddenBics.txt")
Progress: 100%|████████████████████| Time: 0:00:27 (13.65 ms/it)
Dict{String, Any} with 6 entries:
  "recovery"       => 1.0
  "relevance"      => 1.0
  "ce"             => 1.0
  "biclusters"     => [Dict("rows"=>[201, 202, 203, 204, 2…
  "num_iterations" => 758
  "algorithm_time" => 27.3088
```

For more information check: `?run_ebic`.

## Getting more test data

We provide three ready-to-use datasets which can be fetched from a remote DVC repository on Google Drive:
 - Unibic (69MB)
 - RecBic Maintext (211MB)
 - RecBic Sup (7.8GB)

[DVC](https://dvc.org) needs to be installed on your system to download the data.

```bash
pip install dvc[gdrive]
```

Run `dvc pull` to fetch all three datasets, whereas to download a particular one use `dvc pull <path_to_dataset_dvc>`, e.g., `dvc pull data/unibic.dvc`.

_When using DVC for the first time in a repository, one must authenticate with their Google account following instructions given by DVC._

## Ready-to-use benchmarks

Benchmarks:

 - [Unibic](https://www.nature.com/articles/srep23466)
 - [RecBic](https://doi.org/10.1093/bioinformatics/btaa630):
    - maintext
    - supplement

To test all three datasets at once run the following:

```julia
julia> using EBIC
julia> benchmark_all()
┌ Info: #############################
│ TEST GROUP: 'narrow_100_10'
└ ###################################
┌ Info: #############################
│ Test case  : narrow_100_10_data1.txt
└ Groundtruth: narrow_100_10_data1_hiddenBics.txt
Progress: 100%|████████████████████| Time: 0:00:59
┌ Info: Metrics:
│ Prelic relevance   : 1.0
│ Prelic recovery    : 1.0
└ Clustering error   : 1.0
```

The above is the same as running the three functions one after another:

```julia
benchmark_unibic()
benchmark_recbic_maintext()
benchmark_recbic_sup()
```

The test results are save in `results/EBIC.jl` folder in the repository
root directory by default, a different result path can be specified as an argument
(e.g., `benchmark_unibic(out_dir = "new_results")`).

## Cite us

```
@inproceedings{10.1145/3449726.3463197,
    author = {Renc, Pawe\l{} and Orzechowski, Patryk and Byrski, Aleksander and W\u{a}s, Jaros\l{}aw and Moore, Jason H.},
    title = {EBIC.JL: An Efficient Implementation of Evolutionary Biclustering Algorithm in Julia},
    year = {2021},
    isbn = {9781450383516},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3449726.3463197},
    doi = {10.1145/3449726.3463197},
    booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion},
    pages = {1540–1548},
    numpages = {9},
    keywords = {data mining, biclustering, parallel algorithms, evolutionary computation, machine learning},
    location = {Lille, France},
    series = {GECCO '21}
}

@inproceedings{10.1145/3449726.3462739,
    author = {Renc, Pawe\l{} and Orzechowski, Patryk and Byrski, Aleksander and W\k{a}s, Jaros\l{}aw and Moore, Jason H.},
    title = {Rapid Prototyping of Evolution-Driven Biclustering Methods in Julia},
    year = {2021},
    isbn = {9781450383516},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3449726.3462739},
    doi = {10.1145/3449726.3462739},
    booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion},
    pages = {61–62},
    numpages = {2},
    keywords = {biclustering, evolutionary computation, parallel algorithms, data mining, machine learning},
    location = {Lille, France},
    series = {GECCO '21}
}
```

