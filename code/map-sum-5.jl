function foldl′(step, acc, xs)
    for x in xs
        acc = step(acc, x)
    end
    return acc
end

map′(f, xs) = foldl′(mapping_step(f, push!), xs)
sum′(f, xs) = foldl′(mapping_step(f, +), xs)
