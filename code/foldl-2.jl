struct Reduced{T}
    value::T
end

function foldl′(step, xs; init)
    acc = init
    for x in xs
        acc = step(acc, x)
        if acc isa Reduced
            return acc
        end
    end
    return acc
end
