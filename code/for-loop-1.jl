ans = for x in xs
    A(x)
    B(x) && break
    C(x) && break D(x)
else
    E(x)
end

