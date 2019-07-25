map′(f, xs) = foldl′(mapping_step(f, push!), xs; init=[])
