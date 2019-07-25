function map′(f, xs)
    acc = []
    for x in xs
        # loop body
        acc = push!(acc, f(x))
    end
    return acc
end
