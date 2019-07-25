function filtering_step(pred, step)

    function new_step(acc, x)
        if pred(x)
            acc = step(acc, x)
        end
        return acc
    end

    return new_step
end

function filter′(pred, xs)
    acc = []
    step = filtering_step(pred, push!)
    for x in xs
        acc = step(acc, x)
    end
    return acc
end
