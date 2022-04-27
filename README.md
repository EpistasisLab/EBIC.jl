# EBIC.jl

EBIC is a next-generation biclustering algorithm based on artificial intelligence. EBIC is probably the first algorithm capable of discovering the most challenging patterns (i.e. row-constant, column-constant, shift, scale, shift-scale and trend-preserving) in complex and noisy data with average accuracy of over 90%. It is also one of the very few parallel biclustering algorithms that use at least one graphics processing unit (GPU) and is ready for big-data challenges.

The repository contains the new version of [EBIC](https://github.com/EpistasisLab/ebic) rewritten and improved.

## Requirements

- Julia 1.6 or higher
- CUDA-capable GPU with compute capability 5.0 (Maxwell) or higher

## Setting up project

1. Clone the project.

2. Enter the project root directory.

3. Install dependencies

```bash
julia --project -E "using Pkg; Pkg.instantiate()"
```

4. Start quick test on `data/example_input.json` (running a Julia program takes siginificant amount of time because source code has to be compiled each time, Julia REPL is a recommended way of doing it).

```bash
julia --project src/EBIC.jl -v -i data/example_input.csv
```

## Getting test data

We provide three ready-to-use datasets which can be fetched from remote DVC repository on Google Drive:
 - Unibic (69MB)
 - RecBic Maintext (211MB)
 - RecBic Sup (7.8GB)

[DVC](https://dvc.org) needs to be installed on your system to download the data.

```bash
pip install dvc[gdrive]
```

Run inside the repository `dvc pull` if you want to fetch all three datasets, whereas to download a particular one use `dvc pull <dataset_name>` where `dataset_name` is one of the following:
- `unibic`
- `recbic_maintext`
- `recbic_sup`

_When using DVC for the first time in a repository, one must authenticate with their Google account following instructions given by DVC._

## Usage

### Julia REPL

_This is the recommanded way of testing Julia applications._

#### Running the algorithm

The algorithm is run using `run_ebic()`. The function shares the same API as the command line version described below.
The example is run with extended results (`best_bclrs_stats`):

```julia
julia> using EBIC
julia> run_ebic("data/example_input.csv")
Progress: 100%|████████████████████| Time: 0:01:03
Dict{String, Any} with 5 entries:
  "data_load_time"      => 16.8467
  "biclusters"          => [Dict("rows"=>[31, 32, 33, 34, 35,…
  "last_iter_tabu_hits" => 305
  "algorithm_time"      => 63.0405
  "performed_iters"     => 744
```

#### Benchmarking the algorithm on the provided datasets

To test all three datasets at once run the following:

```julia
julia> include("test/Synthtest.jl"); using .synthtest
julia> test_all()
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
│ Clustering error   : 1.0
└ Last iter tabu hits: 331
```

The above is the same as running the three functions one after another:

```julia
test_unibic()
test_recbic_maintext()
test_recbic_sup()
```

The test results are save in `results/EBIC.jl` folder in the repository 
root directory by default, a different result path can be specified as an argument
(e.g., `test_unibic(out_dir = "new_results")`).

### Command line

```
usage: EBIC.jl [-i INPUT_PATH] [-n MAX_ITERATIONS] [-b MAX_BICLUSTERS]
               [-x OVERLAP_THRESHOLD] [-t] [-g GPUS_NUM]
               [-a APPROX_TRENDS_RATIO] [-v] [-s] [-o] [-h]

EBIC is a next-generation biclustering algorithm based on artificial
intelligence (AI). EBIC is probably the first algorithm capable of
discovering the most challenging patterns (i.e. row-constant,
column-constant, shift, scale, shift-scale and trend-preserving) in
complex and noisy data with average accuracy of over 90%. It is also
one of the very few parallel biclustering algorithms that use at least
one graphics processing unit (GPU) and is ready for big-data
challenges.

optional arguments:
  -i, --input INPUT_PATH
                        The path to the input file. (default:
                        "data/example_input.csv")
  -n, --max_iterations MAX_ITERATIONS
                        The maximum number of iterations of the
                        algorithm. (type: Int64, default: 2000)
  -b, --biclusters_num MAX_BICLUSTERS
                        The number of biclusters that will be returned
                        in the end. (type: Int64, default: 3)
  -x, --overlap_threshold OVERLAP_THRESHOLD
                        The maximum similarity level of each two
                        chromosomes held in top rank list. (type:
                        Float64, default: 0.75)
  -t, --negative_trends
                        Enable negative trends.
  -g, --gpus_num GPUS_NUM
                        The number of gpus the algorithm should run
                        on. (type: Int64, default: 1)
  -a, --approx_trends APPROX_TRENDS_RATIO
                        (type: Float32, default: 0.85)
  -v, --verbose         Turn on the progress bar.
  -s, --best_bclrs_stats
                        Evaluate resulting biclusters finding
                        iteration and time. Enabled, it slightly
                        worsens overall algorithm performance.
  -o, --output          Save biclusters to a file in the JSON format.
                        The output file name is a concatenation of the
                        input file name and '-res.json' suffix.
  -h, --help            show this help message and exit
```

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

