function sum′(f, xs)
    acc = 0
    step = mapping_step(f, +)
    for x in xs
        acc = step(acc, x)
    end
    return acc
end
