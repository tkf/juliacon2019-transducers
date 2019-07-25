sum′(f, xs) = foldl′(mapping_step(f, +), xs; init=0)
map′(f, xs) = foldl′(mapping_step(f, push!), xs; init=[])
