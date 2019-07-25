function takingwhile(f)
    function transducer(step)
        function new_step(acc, x)
            if f(x)
                return step(acc, f(x))
            else
                # What to do here?
            end
        end
        return new_step
    end
    return transducer
end
