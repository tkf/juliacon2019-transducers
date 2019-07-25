function map′(f, xs)
    acc = []
    step = mapping_step(f, push!)
    for x in xs
        acc = step(acc, x)
    end
    return acc
end
