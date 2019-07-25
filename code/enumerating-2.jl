function enumerating(step)
    function new_step((count, acc), x)
        count += 1
        return (count, step(acc, (count, x)))
    end
    return new_step
end

init = (0, [])
foldl′(enumerating(push!), [:a, :b, :c]; init=init)
