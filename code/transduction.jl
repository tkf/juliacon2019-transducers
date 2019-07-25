using Transducers
using Transducers: R_, wrap, Unseen, xform, inner, wrapping, wrap, unwrap
import Transducers: start, next, complete

struct TransductionBuffer
    buffer::Vector{Any}
end

Base.iterate(itr::TransductionBuffer, ::Any = nothing) =
    if isempty(itr.buffer)
        nothing
    else
        popfirst!(itr.buffer), nothing
    end

struct Transduction{T} <: Transducer
    ixf::T
end

function start(rf::R_{Transduction}, result)
    buffer = []
    itr = xform(rf).ixf(TransductionBuffer(buffer))
    return wrap(rf, (itr, Unseen(), buffer), start(inner(rf), result))
end

next(rf::R_{Transduction}, result, input) =
    wrapping(rf, result) do (itr, itstate0, buffer), result
        push!(buffer, input)
        if itstate0 isa Unseen
            ret = iterate(itr)
        else
            ret = iterate(itr, itstate0)
        end
        while true
            if ret === nothing
                return (itr, itstate0, buffer), reduced(complete(inner(rf), result))
            end
            itout, itstate = ret
            result = next(inner(rf), result, itout)
            if result isa Reduced || isempty(buffer)
                return (itr, itstate, buffer), result
            end
            ret = iterate(itr, itstate)
        end
    end

complete(rf::R_{Transduction}, result) =
    complete(inner(rf), unwrap(rf, result)[2])

@show collect(Transduction(itr -> Base.Generator(x -> x + 1, itr)), 1:2)
@show collect(Transduction(itr -> Iterators.filter(isodd, itr)), 1:2)
