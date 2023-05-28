module pandprep

using Mimi
using Distributions
using Roots
using DataFrames
using CSV
using VegaLite

export construct_model

const model_time = collect(2000:2500)

time_range(from::Int, to::Int) = TimestepIndex(from):TimestepIndex(to)

function u(consumption, risk_aversion, critical_level)
    consumption = float(consumption)
    utility = (
        consumption^(1 - risk_aversion) / (1 - risk_aversion)
        - critical_level^(1 - risk_aversion) / (1 - risk_aversion)
    )
    return utility
end

function f(B, theta, mu_max, rho, A, N_max, gamma, c_bar, delta, beta, Delta)
    z = (
            theta * B^(theta - 1) * (mu_max / (1 + B^theta)^2)
            / (mu_max / (1 + B^theta) + rho)
    )
    u_hat = (
        u(A - B / N_max, gamma, c_bar)
        - u(A, gamma, c_bar) * (
            (1 - delta)^beta * (1 - exp(-rho * Delta)) + exp(-rho * Delta)
        )
    )
    image = z * u_hat - (1 / N_max) * (A - B / N_max)^(-gamma)
    return image
end


@defcomp population begin
    N = Variable(index = [time])

    N_max = Parameter()
    pandemic = Parameter(index = [time])
    pandemic_mortality = Parameter()
    generation_span = Parameter{Int}()

    function run_timestep(p, v, d, t)
        if is_first(t)
            v.N[t] = p.N_max
            return
        end

        v.N[t] =  v.N[t - 1]

        # If there is a pandemic, a share of the population dies
        if p.pandemic[t] == 1
            v.N[t] -= v.N[t] * p.pandemic_mortality
        end

        # If there was a pandemic some years ago, the victims come back
        # into the population
        if t.t == 1 + p.generation_span && p.pandemic[t - p.generation_span] == 1
            v.N[t] += p.N_max * p.pandemic_mortality
        elseif t.t > 1 + p.generation_span && p.pandemic[t - p.generation_span] == 1
            v.N[t] += v.N[t - p.generation_span - 1] * p.pandemic_mortality
        end

        # Check that population is positive and does not exceed the maximum
        if v.N[t] > p.N_max
            v.N[t] = p.N_max
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
        v.C[t] = v.Y[t] - p.B[t]
        v.c[t] = v.C[t] / p.N[t]
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

    function run_timestep(p, v, d, t)
        v.W[t] = p.N[t]^p.beta * u(p.c[t], p.gamma, p.c_bar)

        utility_discount_factors = [exp(-p.rho * date) for date in 0:(t.t - 1)]
        v.W_intertemporal[t] = sum(utility_discount_factors .* v.W[time_range(1, t.t)])
    end
end


@defcomp pandemic_risk begin
    mu = Variable(index = [time])         # pandemic hazard rate
    pandemic = Variable(index = [time])   # indicator of whether there is a pandemic

    B = Parameter(index = [time])   # prevention
    theta = Parameter()             # decreasing effectiveness of prevention
    mu_max = Parameter()            # maximum hazard rate

    mu_first = Parameter()        # intial pandemic hazard rate
    multiple = Parameter{Bool}()  # whether there are multiple pandemics

    function run_timestep(p, v, d, t)
        if is_first(t)
            v.mu[t] = p.mu_first
        else
            v.mu[t] = p.mu_max / (1 + p.B[t-1]^p.theta)
        end

        if is_first(t) || p.multiple || !any(v.pandemic[time_range(1, t.t - 1)] .== 1)
            hazard = Bernoulli(v.mu[t])
            v.pandemic[t] = rand(hazard, 1)[1]
        else
            v.pandemic[t] = 0
        end
    end
end


@defcomp policy begin
    B = Variable(index = [time])         # prevention

    pandemic = Parameter(index = [time])  # pandemic history
    constant_prevention = Parameter()     # pre-pandemic level of prevention
    multiple = Parameter{Bool}()          # whether there are multiple pandemics

    function run_timestep(p, v, d, t)
        if is_first(t) || p.multiple || all(p.pandemic[time_range(1, t.t - 1)] .== 0)
            v.B[t] = p.constant_prevention
        else
            v.B[t] = 0
        end
    end
end


function construct_model(B, parameters::Dict)
    model = Model()

    set_dimension!(model, :time, model_time)

    add_comp!(model, policy)
    add_comp!(model, pandemic_risk)
    add_comp!(model, population)
    add_comp!(model, economy)
    add_comp!(model, welfare)

    update_param!(model, :policy, :constant_prevention, B)
    update_param!(model, :policy, :multiple, parameters["multiple"])
    update_param!(model, :pandemic_risk, :mu_first, parameters["mu_first"])
    update_param!(model, :pandemic_risk, :mu_max, parameters["mu_max"])
    update_param!(model, :pandemic_risk, :theta, parameters["theta"])
    update_param!(model, :pandemic_risk, :multiple, parameters["multiple"])
    update_param!(model, :population, :N_max, parameters["N_max"])
    update_param!(model, :population, :pandemic_mortality, parameters["pandemic_mortality"])
    update_param!(model, :population, :generation_span, parameters["generation_span"])
    update_param!(model, :economy, :A, parameters["A"])
    update_param!(model, :welfare, :gamma, parameters["gamma"])
    update_param!(model, :welfare, :c_bar, parameters["c_bar"])
    update_param!(model, :welfare, :beta, parameters["beta"])
    update_param!(model, :welfare, :rho, parameters["rho"])


    connect_param!(model, :pandemic_risk, :B, :policy, :B)
    connect_param!(model, :economy, :B, :policy, :B)
    connect_param!(model, :population, :pandemic, :pandemic_risk, :pandemic)
    connect_param!(model, :policy, :pandemic, :pandemic_risk, :pandemic)
    connect_param!(model, :economy, :N, :population, :N)
    connect_param!(model, :welfare, :N, :population, :N)
    connect_param!(model, :welfare, :c, :economy, :c)
    connect_param!(model, :welfare, :c, :economy, :c)

    return model
end


function run_model(B, parameters)
    model = construct_model(B, parameters)
    run(model)
    prevention = B
    pandemic_time = findfirst(pandemic -> pandemic == 1, model[:pandemic_risk, :pandemic])
    welfare = model[:welfare, :W_intertemporal] |> last
    return prevention, pandemic_time, welfare
end


function run_model_several_times(B::Number, parameters, n)
    return (x -> run_model(x, parameters)).([B for _ in 1:n])
end


function run_model_several_times(B::Array, parameters, n_each)
    values_to_run_over = vcat([[b for _ in 1:n_each] for b in B]...)
    return (x -> run_model(x, parameters)).(values_to_run_over)
end


function run_and_summarise(B, parameters, n)
    df = run_model_several_times(B, parameters, n) |> DataFrame
    df = rename(df, [:prevention, :pandemic_time, :welfare])
    df = groupby(df, :prevention)
    df = combine(
        df,
        nrow => :n_runs,
        [:pandemic_time, :welfare] .=> [minimum maximum median mean]
    )
end


function plot_welfare_vs_prevention(simulations_df::DataFrame)
    y_min = simulations_df.welfare_mean |> minimum
    y_max = simulations_df.welfare_mean |> maximum
    x_min = simulations_df.prevention |> minimum
    x_max = simulations_df.prevention |> maximum
    x_argmax = simulations_df[argmax(simulations_df.welfare_mean), :prevention]

    plot = simulations_df |> @vlplot(
        mark={:line, point=true, color="#999", strokeWidth=1},
        x={
            :prevention,
            title="Prevention",
            axis={offset=7, values=[x_min, x_argmax, x_max]}
        },
        y={
            :welfare_mean,
            title="Average welfare",
            scale={domain=[y_min, y_max], nice=false},
            axis={values=[y_min, y_max], offset=10, format="d"}
        },
        config={
            view={width=500, height=250, stroke=nothing},
            axisY={titleAngle=0, titleX=-25, titleY=-10, titleAlign="left"},
            axis={grid=false}
        }
    )
    return plot
end


function plot_welfare_vs_prevention(path_to_simulations_df::String)
    simulations_df = CSV.File(path_to_simulations_df) |> DataFrame
    return plot_welfare_vs_prevention(simulations_df)
end


"Optimal level of prevention according to the analytical model"
B_star = find_zero(B -> f(B, 0.5, 0.2, 0.01, 8, 10, 2, 1, 0.4, 1, 25), 2)

default_parameters = Dict(
    "multiple" => false,
    "mu_first" => 0.0,
    "mu_max" => 0.2,
    "theta" => 0.5,
    "N_max" => 10,
    "pandemic_mortality" => 0.4,
    "generation_span" => 25.0,
    "A" => 8.0,
    "gamma" => 2.0,
    "c_bar" => 1.0,
    "beta" => 1.0,
    "rho" => 0.01,
)

function run_and_save_simulation(prevention_values, parameters, n_each)
    n_pandemics = parameters["multiple"] ? "multiple_pandemics" : "one_pandemic"
    save_path = joinpath("data", "simulations_$(n_pandemics)_$(n_each)_runs.csv")
    simulations_df = run_and_summarise(prevention_values, parameters, n_each)
    CSV.write(save_path, simulations_df)
    return nothing
end

run_and_save_simulation([0, 5, 8, 9.46, 10, 11, 15, 20, 30, 50], default_parameters, 500)
plot_welfare_vs_prevention(joinpath("data", "simulations_one_pandemic_500_runs.csv")) |>
    save(joinpath("images", "one_pandemic_500_runs.svg"))

run_and_save_simulation([5, 8, B_star, 10, 11, 15], default_parameters, 5000)
plot_welfare_vs_prevention(joinpath("data", "simulations_one_pandemic_5000_runs.csv")) |>
    save(joinpath("images", "one_pandemic_5000_runs.svg"))

end # module pandprep
