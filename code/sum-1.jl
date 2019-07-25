function sum′(f, xs)
    sum = 0
    for x in xs
        sum += f(x)
    end
    return sum
end
