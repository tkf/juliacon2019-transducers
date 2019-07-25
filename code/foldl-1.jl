function foldl′(step, xs; init)
    acc = init
    for x in xs
        acc = step(acc, x)
    end
    return acc
end
