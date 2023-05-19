module pandprep

using Mimi

export construct_model

const model_time = 0:500

@defcomp population begin
    N = Variable(index = [time])

    N_max = Parameter()
    pandemic = Parameter(index = [time])
    pandemic_mortality = Parameter()
    generation_span = Parameter()

    function run_timestep(p, v, d, t)
        v.N[t] = v.N[t - 1]

        # If there is a pandemic, a share of the population dies
        if p.pandemic[t]
            v.N[t] -= v.N[t] * p.pandemic_mortality
        end

        # If there was a pandemic some years ago, the victims come back
        # into the population
        if p.pandemic[t - generation_span]
            v.N[t] += v.N[t - generation_span - 1] * p.pandemic_mortality
        end

        # Check that population does is positive and does not exceed the maximum
        if v.N[t] > p.N_max
            v.N[t] = N_max
        end
        if v.N[t] < 0
            v.N[t] = 0
        end
    end
end


@defcomp economy begin
    Y = Variable(index = [time])   # production
    C = Variable(index = [time])   # consumption
    c = Variable(index = [time])   # per-capita consumption

    N = Parameter(index = [time])  # population
    B = Parameter(index = [time])  # prevention
    A = Parameter()                # technology

    function run_timestep(p, v, d, t)
        v.Y[t] = p.A * p.N[t]
        v.C[t] = v.Y - p.B[t]
        v.c[t] = v.Y / p.N[t]
    end
end


@defcomp welfare begin
    W = Variable(index = [time])                # welfare at time t
    W_intertemporal = Variable(index = [time])  # intertemporal welfare

    N = Parameter(index = [time])  # population
    c = Parameter(index = [time])  # per-capita consumption
    gamma = Parameter()            # coefficient of relative aversion
    c_bar = Parameter()            # critical level of utility
    beta = Parameter()             # population ethics parameter
    rho = Parameter()              # utility discount rate

    function utility(consumption::AbstractFloat, risk_aversion, critical_level)
        return (
            consumption^(1 - risk_aversion) / (1 - risk_aversion)
            - critical_level^(1 - risk_aversion) / (1 - risk_aversion)
            )
        end

    function run_timestep(p, v, d, t)
        v.W[t] = p.N[t]^beta * utility(p.c[t], p.gamma, p.c_bar)

        utility_discount_factors = [exp(-p.rho * date) for date in 0:t]
        v.W_intertemporal[t] = sum(utility_discount_factors * v.W[0:t])
    end
end


function construct_model()
    model = Model()

    set_dimension!(model, :time, model_time)

    add_comp!(model, population)

    return model
end

end # module pandprep
