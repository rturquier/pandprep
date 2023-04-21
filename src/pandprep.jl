module pandprep

using Mimi

export construct_model

const model_time = 2023:2523

function construct_model()
    model = Model()

    set_dimension!(model, :time, model_time)

    return model
end

end # module pandprep
