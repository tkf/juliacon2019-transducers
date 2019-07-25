mapping(f) = step -> (acc, x) -> step(acc, f(x))
filtering(pred) = step -> (acc, x) -> pred(x) ? step(acc, x) : acc

xf = filtering(!ismissing) ∘
    mapping(x -> x - 1) ∘
    filtering(x -> x > 0)

xs = [missing, 2, 0, missing, 3]
foldl′(xf(+), xs; init=0) == 3
foldl′(xf(push!), xs; init=[]) == [1, 2]
