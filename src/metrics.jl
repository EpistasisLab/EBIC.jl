module metrics

export eval_metrics

using Munkres: munkres
using CSV
using Tables

function eval_metrics(biclusters::Vector, input_path::String, ground_truth::Vector)
    ground_truth = deepcopy(ground_truth)

    for bclr in ground_truth
        bclr["cols"] .+= 1
        bclr["rows"] .+= 1
    end

    dataset = Tables.matrix(CSV.File(input_path; drop=[1], header=false, skipto=2))
    nrows, ncols = size(dataset)

    relevance = prelic_relevance(biclusters, ground_truth)
    recovery = prelic_recovery(biclusters, ground_truth)
    ce = clustering_error(biclusters, ground_truth, nrows, ncols)

    return relevance, recovery, ce
end

"""
Based on: https://github.com/padilha/biclustlib/blob/master/biclustlib/evaluation/prelic.py
"""
function prelic_relevance(predicted_biclusters, reference_biclusters)::Float64
    col_score = _match_score(predicted_biclusters, reference_biclusters, "cols")
    row_score = _match_score(predicted_biclusters, reference_biclusters, "rows")

    return sqrt(col_score * row_score)
end

function prelic_recovery(predicted_biclusters, reference_biclusters)::Float64
    return prelic_relevance(reference_biclusters, predicted_biclusters)
end

function _match_score(predicted_biclusters, reference_biclusters, attr)::Float64
    isempty(predicted_biclusters) && isempty(reference_biclusters) && return 1
    isempty(predicted_biclusters) || isempty(reference_biclusters) && return 0

    return sum([
        maximum([
            length(intersect(bp[attr], br[attr])) / length(union(bp[attr], br[attr])) for
            br in reference_biclusters
        ]) for bp in predicted_biclusters
    ]) / length(predicted_biclusters)
end

"""
Based on: https://github.com/padilha/biclustlib/blob/bfadbd6944cc3b5b297e4018dd1437ddfb1f589f/biclustlib/evaluation/subspace.py#L28
"""
function clustering_error(predicted_biclusters, reference_biclusters, nrows, ncols)::Float64
    isempty(predicted_biclusters) && isempty(reference_biclusters) && return 1
    isempty(predicted_biclusters) || isempty(reference_biclusters) && return 0

    union_size = _calc_size(
        predicted_biclusters, reference_biclusters, nrows, ncols, "union"
    )
    dmax = _calc_dmax(predicted_biclusters, reference_biclusters)

    return dmax / union_size
end

function _calc_size(
    predicted_biclusters, reference_biclusters, nrows, ncols, operation
)::Float64
    pred_count = _count_biclustering(predicted_biclusters, nrows, ncols)
    true_count = _count_biclustering(reference_biclusters, nrows, ncols)

    if operation == "union"
        return sum(broadcast(max, pred_count, true_count))
    elseif operation == "intersection"
        return sum(broadcast(min, pred_count, true_count))
    end

    throw(ArgumentError("Incorrect argument: $operation"))
end

function _count_biclustering(biclusters, nrows, ncols)
    count = zeros(Int, nrows, ncols)

    for b in biclusters
        for (x, y) in Base.product(b["rows"], b["cols"])
            count[x, y] += 1
        end
    end
    return count
end

function _calc_dmax(predicted_biclusters, reference_biclusters)::Float64
    pred_sets = _bic2sets(predicted_biclusters)
    true_sets = _bic2sets(reference_biclusters)

    cost_matrix = [
        maxintfloat() - length(intersect(b, g)) for g in true_sets, b in pred_sets
    ]
    indices = munkres(cost_matrix)

    return sum([maxintfloat() - cost_matrix[x, y] for (x, y) in enumerate(indices)])
end

function _bic2sets(biclusters)
    return [Set(Base.product(b["rows"], b["cols"])) for b in biclusters]
end

end
