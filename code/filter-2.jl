filter′(f, xs) = foldl′(
    filtering_step(f, push!),
    xs, init=[])

function filtering_step(f, step)

    function new_step(acc, x)
        if f(x)
            acc = step(acc, x)
        end
        return acc
    end

    return new_step
end

filtering_step(f, step) =
    (acc, x) ->
        f(x) ? step(acc, x) : acc
# (equivalent)
