# EBIC.jl

EBIC is a next-generation biclustering algorithm based on artificial intelligence. EBIC is probably the first algorithm capable of discovering the most challenging patterns (i.e. row-constant, column-constant, shift, scale, shift-scale and trend-preserving) in complex and noisy data with average accuracy of over 90%. It is also one of the very few parallel biclustering algorithms that use at least one graphics processing unit (GPU) and is ready for big-data challenges.

The repository contains the new version of [EBIC](github.com/athril/ebic) rewritten and improved.

## Requirements

- Julia 1.5 or higher
- CUDA-capable GPU with compute capability 5.0 (Maxwell) or higher

## Setting project

1. Clone the project.
```bash
git clone https://github.com/prenc/EBIC.jl.git
```

2. Enter the project directory:
```bash
cd EBIC.jl
```

3. Install dependencies
```bash
julia --project -E "using Pkg; Pkg.instantiate()"
```

4. Start quick test on `data/example_input.json`.
```bash
julia --project src/Ebic.jl
```

## Getting test data

We provide three processed test datasets which can be fetched from remote DVC repository on Google Drive:
 - Unibic
 - RecBic Maintext
 - RecBic Sup (much larger the others)
 
In order to aquire the datasets, the [DVC](https://dvc.org) application is requried installed on your system.

Run inside the repository `dvc pull` if all three datasets are to be downloaded, whereas to fetch the particular one use `dvc pull <dataset_name>` where `dataset_name` is one of the following:
- `unibic`
- `recbic_maintext`
- `recbic_sup`

_When using DVC for the first time for the repository, you how to authenticate your Google account following instructions given by DVC._

## Usage

### Julia REPL

_This is the recommanded way of testing Julia applications._

#### Running the algorithm

The algorithm is run using `run_ebic()`. The function shares the same API as command line version described below. Example run with extended results (`best_bclrs_stats`):
```julia
julia> include("src/Ebic.jl"); res = Ebic.run_ebic("data/example_input.csv", verbose = true, best_bclrs_stats = true)
Progress: 100%|████████████████████| Time: 0:00:41
Dict{String,Any} with 7 entries:
  "data_load_time"      => 0.0338261
  "best_bclrs_iter"     => 815
  "biclusters"          => [Dict("rows"=>[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, …
  "last_iter_tabu_hits" => 333
  "best_bclrs_time"     => 37.6397
  "algorithm_time"      => 41.164
  "performed_iters"     => 894
```

#### Running tests on provided datasets

To test all three datasets at once run the following:
```julia
julia> include("test/ebic_synth_test.jl"); synthtest.main()
results/Ebic.jl/unibic
####################################
Starting test case: 'narrow_100_10'
####################################
##################################
Testing: 'narrow_100_10_data1.txt'
Ground truth: 'narrow_100_10_data1_hiddenBics.txt'
```

The above is the same as running the three functions one after another:
```julia
synthtest.test_unibic()
synthtest.test_recbic_maintext()
synthtest.test_recbic_sup()
```

The tests' results are save in `output` folder in the repository root.

### Command line
```
usage: Ebic.jl [-i INPUT_PATH] [-n MAX_ITERATIONS] [-b MAX_BICLUSTERS]
               [-x OVERLAP_THRESHOLD] [-t] [-g GPUS_NUM]
               [-a APPROX_TRENDS_RATIO] [-v] [-s] [-o] [-h]

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
