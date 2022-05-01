function mutation_swap(chromo::Chromo)::Chromo
    random_idx1 = rand(1:length(chromo))
    random_idx2 = rand(1:length(chromo))

    while random_idx1 == random_idx2
        random_idx2 = rand(1:length(chromo))
    end

    chromo = copy(chromo)
    chromo[random_idx1], chromo[random_idx2] = chromo[random_idx2], chromo[random_idx1]
    return chromo
end

function mutation_substitution(chromo::Chromo, cols_number::Int)::Chromo
    random_col = rand(1:cols_number)
    while random_col in chromo
        random_col = rand(1:cols_number)
    end
    random_substitution_point = rand(1:length(chromo))
    chromo = copy(chromo)
    chromo[random_substitution_point] = random_col
    return chromo
end

function mutation_insertion(chromo::Chromo, cols_number::Int)::Chromo
    random_col = rand(1:cols_number)
    while random_col in chromo
        random_col = rand(1:cols_number)
    end
    random_insertion_point = rand(1:(length(chromo) + 1))
    return insert!(copy(chromo), random_insertion_point, random_col)
end

function mutation_deletion(chromo::Chromo)::Chromo
    deletion_point = rand(1:length(chromo))
    return deleteat!(copy(chromo), deletion_point)
end

function crossover(chromo1::Chromo, chromo2::Chromo)::Chromo
    cut1_idx = rand(1:length(chromo1))
    cut2_idx = rand(1:length(chromo2))
    new_chromo = chromo1[1:cut1_idx]
    for col_number in chromo2[cut2_idx:end]
        if !(col_number in new_chromo)
            push!(new_chromo, col_number)
        end
    end
    return new_chromo
end
