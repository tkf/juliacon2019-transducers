function filter′(f, xs)
    ys = []
    for x in xs
        if f(x)
            push!(ys, x)
        end
    end
    return ys
end
