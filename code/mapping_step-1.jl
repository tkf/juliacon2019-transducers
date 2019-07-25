function mapping_step(f, step)

    function new_step(acc, x)
        acc = step(acc, f(x))
        return acc
    end

    return new_step
end
