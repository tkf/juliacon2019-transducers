function takingwhile(f)
    function transducer(step)
        function new_step(acc, x)
            if f(x)
                return step(acc, f(x))
            else
                return Reduced(acc)
            end
        end
        return new_step
    end
    return transducer
end

# equivalently:
takingwhile(f) = step -> (acc, x) ->
    f(x) ? step(acc, x) : Reduced(acc)
