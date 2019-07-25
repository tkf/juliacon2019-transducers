# → equivalently:
mapping_step(f, step) =
    (acc, x) -> step(acc, f(x))
