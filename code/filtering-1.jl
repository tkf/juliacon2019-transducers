function filtering(pred)
    function transducer(step)
        function new_step(acc, x)
            if pred(x)
                acc = step(acc, x)
            end
            return acc
        end
        return new_step
    end
    return transducer
end

# equivalently:
filtering(pred) =
    step -> (acc, x) -> pred(x) ? step(acc, x) : acc
