function flattening(step)
    function new_step(acc, xs)
        for x in xs
            acc = step(acc, x)
        end
        return x
    end
    return new_step
end

flattening(step) = (acc, xs) -> foldl′(step, xs)

xf = mapping(x -> 1:x) ∘ flattening
foldl′(xf(push!), 1:3, init=[]) == [1, 1, 2, 1, 2, 3]
