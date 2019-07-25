function map′(f, xs)
    ys = []
    for x in xs
        push!(ys, f(x))
    end
    return ys
end
