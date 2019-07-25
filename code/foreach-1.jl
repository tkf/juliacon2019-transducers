foreach′(f, xs) = foldl′(xs; init=nothing) do _, x
    f(x)
end

foreach′(xs) do x
    println("Got:", x)
end
