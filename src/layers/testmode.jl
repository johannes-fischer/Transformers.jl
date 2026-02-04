StructWalk.constructor(::Type{LayerStyle}, l::WithArg{names}) where names = WithArg{names}
StructWalk.constructor(::Type{LayerStyle}, l::WithOptArg{names, opts}) where {names, opts} = WithOptArg{names, opts}
StructWalk.constructor(::Type{LayerStyle}, l::RenameArgs{new_names, old_names}) where {new_names, old_names} =
    RenameArgs{new_names, old_names}
StructWalk.constructor(::Type{LayerStyle}, l::Branch{target, names}) where {target, names} = Branch{target, names}
StructWalk.constructor(::Type{LayerStyle}, l::Parallel{names}) where names = Parallel{names}

# Check if x only matches the generic set_dropout(x, p) fallback (no specific overload).
# When multiple methods match (e.g. set_dropout(::Flux.Dropout, p) and
# set_dropout(::Flux.Dropout, ::Nothing)), a specific overload exists → return false.
function no_dropout_overload(x)
    ms = methods(set_dropout, Tuple{typeof(x), Any}).ms
    length(ms) == 1 || return false
    ms[].sig.types[2] >: Any # return true only if the type of the first argument of the found method is Any, i.e., if the set_dropout(x, p) in line 17 is found as the only matching method
end

set_dropout(x, p) = postwalk(LayerStyle, x) do xi
    no_dropout_overload(xi) ? xi : set_dropout(xi, p)
end
set_dropout(dp::DropoutLayer, p) = DropoutLayer(dp.layer, p)
set_dropout(dp::Flux.Dropout, p) = Flux.Dropout(p, dp.dims, dp.active, dp.rng)
set_dropout(dp::Flux.Dropout, ::Nothing) = Flux.Dropout(dp.p, dp.dims, false, dp.rng)

no_dropout(x) = set_dropout(x, nothing)

"""
    testmode(model)

Creating a new model sharing all parameters with `model` but used for testing. Currently this is just
 [`no_dropout`](@ref).
"""
testmode(x) = no_dropout(x)

"""
    set_dropout(model, p)

Creating a new model sharing all parameters with `model` but set all dropout probability to `p`.
"""
set_dropout

"""
    no_dropout(model)

Creating a new model sharing all parameters with `model` but disable all dropout.
"""
no_dropout
