function sum′(f, xs)
    acc = 0  # accumulator
    for x in xs
        acc += f(x)
    end
    return acc
end
