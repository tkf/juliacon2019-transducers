function filter′(pred, xs)
    acc = []

    function step(acc, x)
        if pred(x)
            acc = push!(acc, x)
        end
        return acc
    end

    for x in xs
        acc = step(acc, x)
    end
    return acc
end
