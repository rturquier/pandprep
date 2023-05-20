module pandprep

using Mimi
using Distributions

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


function u(consumption::AbstractFloat, risk_aversion, critical_level)
    utility = (
        consumption^(1 - risk_aversion) / (1 - risk_aversion)
        - critical_level^(1 - risk_aversion) / (1 - risk_aversion)
    )
    return utility
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

    function run_timestep(p, v, d, t)
        v.W[t] = p.N[t]^beta * u(p.c[t], p.gamma, p.c_bar)

        utility_discount_factors = [exp(-p.rho * date) for date in 0:t]
        v.W_intertemporal[t] = sum(utility_discount_factors * v.W[0:t])
    end
end


@defcomp pandemic_risk begin
    mu = Variable(index = [time])         # pandemic hazard rate
    pandemic = Variable(index = [time])   # indicator of whether there is a pandemic

    B = Parameter(index = [time])   # prevention
    mu_bar = Parameter()            # maximum hazard rate

    function run_timestep(p, v, d, t)
        v.mu[t] = p.mu_bar / (1 + p.B)

        hazard = Bernoulli(v.mu[t])
        v.pandemic[t] = rand(hazard, 1)
    end
end


function construct_model()
    model = Model()

    set_dimension!(model, :time, model_time)

    add_comp!(model, pandemic_risk)
    add_comp!(model, population)
    add_comp!(model, economy)
    add_comp!(model, welfare)

    update_param!(model, :pandemic_risk, :mu_bar, 0.1)
    update_param!(model, :population, :N_max, 1000)
    update_param!(model, :population, :pandemic_mortality, 0.3)
    update_param!(model, :population, :generation_span, 25.0)
    update_param!(model, :economy, :A, 8.0)
    update_param!(model, :welfare, :gamma, 2.0)
    update_param!(model, :welfare, :c_bar, 1.0)
    update_param!(model, :welfare, :beta, 1.0)
    update_param!(model, :welfare, :rho, 0.01)

    connect_param!(model, :population, :pandemic, :pandemic_risk, :pandemic)
    connect_param!(model, :economy, :N, :population, :N)
    connect_param!(model, :welfare, :N, :population, :N)
    connect_param!(model, :welfare, :c, :economy, :c)
    connect_param!(model, :welfare, :c, :economy, :c)

    return model
end

end # module pandprep
