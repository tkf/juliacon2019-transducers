function sum′(f, xs)
    acc = 0
    for x in xs
        # loop body
        acc = +(acc, f(x))
    end
    return acc
end
