sum′(f, xs) = foldl′(mapping_step(f, +), xs; init=0)
map′(f, xs) = foldl′(mapping_step(f, push!), xs; init=[])

mapping_step(f, step) = (acc, x) -> step(acc, f(x))

function foldl′(step, xs; init)
    acc = init
    for x in xs
        acc = step(acc, x)
    end
    return acc
end
