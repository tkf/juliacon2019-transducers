function mapping(f)
    function transducer(step)
        function new_step(acc, x)
            return step(acc, f(x))
        end
        return new_step
    end
    return transducer
end

# equivalently:
mapping(f) = step -> (acc, x) -> step(acc, f(x))
