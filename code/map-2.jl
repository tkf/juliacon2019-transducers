function map′(f, xs)
    acc = []  # accumulator
    for x in xs
        push!(acc, f(x))
    end
    return acc
end
