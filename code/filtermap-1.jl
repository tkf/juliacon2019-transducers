make_step(step) =
    filtering_step(
        !ismissing,
        mapping_step(
            x -> x - 1,
            filtering_step(
                x -> x > 0,
                step,
            ),
        ),
    )

xs = [missing, 2, 0, missing, 3]
foldl′(make_step(+), xs; init=0) == 3
foldl′(make_step(push!), xs; init=[]) == [1, 2]
