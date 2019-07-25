xf -> itr -> Channel() do ch
    foldl′(xf(put!), itr, init=ch)
end
