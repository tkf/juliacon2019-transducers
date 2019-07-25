function enumerating(step)
    count = 0
    function new_step(acc, x)
        count += 1
        return step(acc, (count, x))
    end
    return new_step
end

foldl′(enumerating(push!), [:a, :b, :c]; init=[])
