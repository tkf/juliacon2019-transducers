map′(f, xs) = foldl′(mapping_step(f, push!), xs)
sum′(f, xs) = foldl′(mapping_step(f, +), xs)
filter′(pred, xs) = foldl′(filtering_step(pred, push!), xs)
