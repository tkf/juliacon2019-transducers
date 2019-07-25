ans = foreach′(xs) do x
    A(x)
    B(x) && return Reduced(nothing)
    C(x) && return Reduced(D(x))
    x
end |> ifunreduced() do x
    E(x)
end

ifunreduced(f, x::Reduced) = x.value
ifunreduced(f, x) = f(x)
ifunreduced(f) = x -> ifunreduced(f, x)
