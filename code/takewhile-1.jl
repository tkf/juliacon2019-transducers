function takewhile(f, xs)
    acc = []
    for x in xs
        if f(x)
            push!(acc, x)
        else
            break
        end
    end
    return acc
end
