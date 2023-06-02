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

function expected_welfare(B, parameters)
    expected_welfare = (
        parameters["N_max"]^parameters["beta"]
        / (
            parameters["mu_max"] / (1 + B^parameters["theta"])
            + parameters["rho"]
        )
        * (
            u(
                parameters["A"] - B / parameters["N_max"],
                parameters["gamma"],
                parameters["c_bar"]
            )
            + (
                1 / parameters["rho"]
                * parameters["mu_max"] / (1 + B^parameters["theta"])
                * u(parameters["A"], parameters["gamma"], parameters["c_bar"])
                * (
                    (1 - parameters["delta"])^parameters["beta"]
                    * (1 - exp(-parameters["rho"] * parameters["Delta"]))
                    + exp(-parameters["rho"] * parameters["Delta"])
                )
            )
        )
    )
    return expected_welfare
end


function plot_expected_welfare(parameters)
    points_x = collect(0:0.01:50)
    points_y = (B -> expected_welfare(B, parameters)).(points_x)
    data = DataFrame([points_x, points_y], ["x", "y"])
    y_min = minimum(points_y)
    y_max = maximum(points_y)
    x_argmax = points_x[argmax(points_y)]

    plot = data |> @vlplot(
        mark=:line,
        x={
                "x:q",
                title="Prevention",
                scale={nice=false},
                axis={offset=7, values=[0, x_argmax, 50], format=".3"}
            },
        y={
            "y:q",
            title="Expected welfare",
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

    N = Parameter(index = [time])  # population
    B = Parameter(index = [time])  # prevention
    A = Parameter()                # technology

    function run_timestep(p, v, d, t)
        v.Y[t] = p.A * p.N[t]
    end
end


@defcomp policy begin
    B = Variable(index = [time])         # prevention
    b = Variable(index = [time])         # prevention as a share of total income

    pandemic = Parameter(index = [time])  # pandemic history
    constant_prevention = Parameter()     # pre-pandemic level of prevention
    multiple = Parameter{Bool}()          # whether there are multiple pandemics

    Y = Parameter(index = [time])  # current population

    function run_timestep(p, v, d, t)
        if p.multiple
            v.b[t] = p.constant_prevention / p.Y[TimestepIndex(1)]
            v.B[t] = v.b[t] * p.Y[t]
        elseif all(p.pandemic[time_range(1, t.t - 1)] .== 0)
            v.B[t] = p.constant_prevention
            v.b[t] = v.B[t] / p.Y[t]
        else
            v.B[t] = 0
            v.b[t] = 0
        end
    end
end


@defcomp welfare begin
    C = Variable(index = [time])                # consumption
    c = Variable(index = [time])                # per-capita consumption
    W = Variable(index = [time])                # welfare at time t
    W_intertemporal = Variable(index = [time])  # intertemporal welfare

    N = Parameter(index = [time])  # population
    Y = Parameter(index = [time])  # production
    B = Parameter(index = [time])  # prevention

    gamma = Parameter()            # coefficient of relative aversion
    c_bar = Parameter()            # critical level of utility
    beta = Parameter()             # population ethics parameter
    rho = Parameter()              # utility discount rate

    function run_timestep(p, v, d, t)
        v.C[t] = p.Y[t] - p.B[t]
        v.c[t] = v.C[t] / p.N[t]
        v.W[t] = p.N[t]^p.beta * u(v.c[t], p.gamma, p.c_bar)

        utility_discount_factors = [exp(-p.rho * date) for date in 0:(t.t - 1)]
        v.W_intertemporal[t] = sum(utility_discount_factors .* v.W[time_range(1, t.t)])
    end
end


function construct_model(B, parameters::Dict)
    model = Model()

    set_dimension!(model, :time, model_time)

    add_comp!(model, pandemic_risk)
    add_comp!(model, population)
    add_comp!(model, economy)
    add_comp!(model, policy)
    add_comp!(model, welfare)

    update_param!(model, :pandemic_risk, :mu_first, parameters["mu_first"])
    update_param!(model, :pandemic_risk, :mu_max, parameters["mu_max"])
    update_param!(model, :pandemic_risk, :theta, parameters["theta"])
    update_param!(model, :pandemic_risk, :multiple, parameters["multiple"])
    update_param!(model, :population, :N_max, parameters["N_max"])
    update_param!(model, :population, :pandemic_mortality, parameters["delta"])
    update_param!(model, :population, :generation_span, parameters["Delta"])
    update_param!(model, :economy, :A, parameters["A"])
    update_param!(model, :policy, :constant_prevention, B)
    update_param!(model, :policy, :multiple, parameters["multiple"])
    update_param!(model, :welfare, :gamma, parameters["gamma"])
    update_param!(model, :welfare, :c_bar, parameters["c_bar"])
    update_param!(model, :welfare, :beta, parameters["beta"])
    update_param!(model, :welfare, :rho, parameters["rho"])

    connect_param!(model, :pandemic_risk, :B, :policy, :B)
    connect_param!(model, :population, :pandemic, :pandemic_risk, :pandemic)
    connect_param!(model, :economy, :B, :policy, :B)
    connect_param!(model, :economy, :N, :population, :N)
    connect_param!(model, :policy, :pandemic, :pandemic_risk, :pandemic)
    connect_param!(model, :policy, :Y, :economy, :Y)
    connect_param!(model, :welfare, :N, :population, :N)
    connect_param!(model, :welfare, :Y, :economy, :Y)
    connect_param!(model, :welfare, :B, :policy, :B)

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
    df = rename(df, [:B, :pandemic_time, :welfare])
    df = groupby(df, :B)
    df = combine(
        df,
        nrow => :n_runs,
        [:pandemic_time, :welfare] .=> [minimum maximum median mean]
    )
    df = transform(df, :B => (x -> x / (parameters["A"] * parameters["N_max"]))  => :b)
end

function convert_prevention_to_share(simulations_df::DataFrame)
    default_technology = 8
    default_population = 10
    default_max_income = default_technology * default_population
    simulations_df[:, "best_prevention"] /= default_max_income
    return simulations_df
end

function plot_simulations(simulations_df::DataFrame; x=:B, y=:welfare_mean)
    if y == :best_prevention
        simulations_df = convert_prevention_to_share(simulations_df)
    end

    y_min = simulations_df[:, y] |> minimum
    y_max = simulations_df[:, y] |> maximum
    x_min = simulations_df[:, x] |> minimum
    x_max = simulations_df[:, x] |> maximum
    x_argmax = simulations_df[argmax(simulations_df[:, y]), x]

    if x in [:b, :rho]
        x_format = "%"
    else
        x_format = "s"
    end

    if x==:rho
        x_title = "Utility discount rate"
        y_title = "Best prevention level"
        x_scale_type = "log"
        y_format = "%"
    else
        x_title = "Prevention"
        y_title = "Expected welfare"
        x_scale_type = "linear"
        y_format = "d"
    end

    plot = simulations_df |> @vlplot(
        mark={:line, point=true, color="#999", strokeWidth=1},
        x={
            x,
            title=x_title,
            scale={nice=false, type=x_scale_type},
            axis={
                offset=7,
                values=[x_min, x_argmax, x_max],
                format=x_format,
                labelFlush=false
            }
        },
        y={
            y,
            title=y_title,
            scale={domain=[y_min, y_max], nice=false},
            axis={values=[y_min, y_max], offset=10, format=y_format}
        },
        config={
            view={width=500, height=250, stroke=nothing},
            axisY={titleAngle=0, titleX=-25, titleY=-10, titleAlign="left"},
            axis={grid=false}
        }
    )
    return plot
end


function plot_simulations(path_to_simulations_df::String; x=:B, y=:welfare_mean)
    simulations_df = CSV.File(path_to_simulations_df) |> DataFrame
    return plot_simulations(simulations_df; x=x, y=y)
end


function run_and_save_simulation(prevention_values, parameters, n_each)
    n_pandemics = parameters["multiple"] ? "multiple_pandemics" : "one_pandemic"
    save_path = joinpath("data", "simulations_$(n_pandemics)_$(n_each)_runs.csv")
    simulations_df = run_and_summarise(prevention_values, parameters, n_each)
    CSV.write(save_path, simulations_df)
    return nothing
end


function get_min_and_max_B_for_top_3_welfare(simulation_df)
    top_n = simulation_df |>
        it -> sort(it, :welfare_mean; rev=true) |>
        it -> first(it, 3)

    B_max = top_n.B |> maximum
    B_min = top_n.B |> minimum

    return [B_min, B_max]
end


function make_range_from_bounds(bounds, step)
    lower_bound = minimum(bounds)
    higher_bound = maximum(bounds)
    range = collect(lower_bound:step:higher_bound)
    return range
end


function get_best_prevention_for_rho(rho::Number, parameters; n_runs_factor=1)
    parameters["rho"] = rho
    n_runs = [10, 50, 100, 500, 500] * n_runs_factor

    precise_run = run_and_summarise(collect(0:10:50), parameters, n_runs[1]) |>
        get_min_and_max_B_for_top_3_welfare |>
        it -> make_range_from_bounds(it, 5) |>
        it -> run_and_summarise(it, parameters, n_runs[2]) |>
        get_min_and_max_B_for_top_3_welfare |>
        it -> make_range_from_bounds(it, 2.5) |>
        it -> run_and_summarise(it, parameters, n_runs[3]) |>
        get_min_and_max_B_for_top_3_welfare |>
        it -> make_range_from_bounds(it, 1) |>
        it -> run_and_summarise(it, parameters, n_runs[4]) |>
        get_min_and_max_B_for_top_3_welfare |>
        it -> make_range_from_bounds(it, 0.5) |>
        it -> run_and_summarise(it, parameters, n_runs[5])

    best_prevention = precise_run.B[argmax(precise_run.welfare_mean)]
    return best_prevention
end


function get_best_prevention_for_rho(rho_values::Array, parameters; n_runs_factor=1)
    B(rho) = get_best_prevention_for_rho(rho, parameters; n_runs_factor=n_runs_factor)
    prevention_values = B.(rho_values)
    return prevention_values
end


function get_best_prevention_for_rho_and_save(
    rho_values, parameters, save_path; n_runs_factor=1
)
    prevention_values = get_best_prevention_for_rho(
        rho_values, parameters; n_runs_factor=n_runs_factor
    )
    df = DataFrame([rho_values, prevention_values], ["rho", "best_prevention"])
    CSV.write(save_path, df)
    return df
end


"Optimal level of prevention according to the analytical model"
B_star = find_zero(B -> f(B, 0.5, 0.2, 0.01, 8, 10, 2, 1, 0.4, 1, 25), 2)

default_parameters = Dict(
    "multiple" => false,
    "mu_first" => 0.0,
    "mu_max" => 0.2,
    "theta" => 0.5,
    "N_max" => 10,
    "delta" => 0.4,
    "Delta" => 25.0,
    "A" => 8.0,
    "gamma" => 2.0,
    "c_bar" => 1.0,
    "beta" => 1.0,
    "rho" => 0.01,
)


function reproduce_simulations()
    run_and_save_simulation(
        [0, 5, 8, 9.46, 10, 11, 15, 20, 30, 50],
        default_parameters,
        500
        )

    run_and_save_simulation([5, 8, B_star, 10, 11, 15], default_parameters, 5000)

    default_parameters_multiple = copy(default_parameters)
    default_parameters_multiple["multiple"] = true
    run_and_save_simulation(
        [0, 5, 10, 15, 20, 25, 30, 50],
        default_parameters_multiple,
        500
    )

    run_and_save_simulation([15, 20, 25, 30, 35], default_parameters_multiple, 5000)

    rho_values = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    save_path = joinpath("data", "prevention_vs_rho_single_pandemic.csv")
    get_best_prevention_for_rho_and_save(rho_values, default_parameters, save_path)

    save_path = joinpath("data", "prevention_vs_rho_multiple_pandemics.csv")
    get_best_prevention_for_rho_and_save(rho_values, default_parameters_multiple, save_path)

    # Previous results were strange. Let's re-run it to see if it changes a lot:
    save_path = joinpath("data", "prevention_vs_rho_multiple_pandemics_2.csv")
    get_best_prevention_for_rho_and_save(rho_values, default_parameters_multiple, save_path)

    # It did change a lot. Let's generate the same data with 10 times more model runs:
    save_path = joinpath("data", "prevention_vs_rho_multiple_pandemics_5000.csv")
    get_best_prevention_for_rho_and_save(
        rho_values,
        default_parameters_multiple,
        save_path;
        n_runs_factor=10
        )

end


function reproduce_plots()
    plot_expected_welfare(default_parameters) |>
        save(joinpath("images", "expected_welfare.svg"))

    plot_simulations(joinpath("data", "simulations_one_pandemic_500_runs.csv")) |>
        save(joinpath("images", "one_pandemic_500_runs.svg"))

    joinpath("data", "simulations_one_pandemic_5000_runs.csv") |>
        plot_simulations |>
        save(joinpath("images", "one_pandemic_5000_runs.svg"))

    joinpath("data", "simulations_multiple_pandemics_500_runs.csv") |>
        (it -> plot_simulations(it; x=:b)) |>
        save(joinpath("images", "multiple_pandemics_500_runs.svg"))

    joinpath("data", "simulations_multiple_pandemics_5000_runs.csv") |>
        (it -> plot_simulations(it; x=:b)) |>
        save(joinpath("images", "multiple_pandemics_5000_runs.svg"))

    # Plot best prevention as a function of rho
    joinpath("data", "prevention_vs_rho_single_pandemic.csv") |>
        it -> plot_simulations(it; x=:rho, y=:best_prevention) |>
        save(joinpath("images", "prevention_vs_rho_single_pandemic.svg"))

    joinpath("data", "prevention_vs_rho_multiple_pandemics.csv") |>
        it -> plot_simulations(it; x=:rho, y=:best_prevention) |>
        save(joinpath("images", "prevention_vs_rho_multiple_pandemics_500_1.svg"))

    joinpath("data", "prevention_vs_rho_multiple_pandemics_2.csv") |>
        it -> plot_simulations(it; x=:rho, y=:best_prevention) |>
        save(joinpath("images", "prevention_vs_rho_multiple_pandemics_500_2.svg"))

    joinpath("data", "prevention_vs_rho_multiple_pandemics_5000.csv") |>
        it -> plot_simulations(it; x=:rho, y=:best_prevention) |>
        save(joinpath("images", "prevention_vs_rho_multiple_pandemics_5000.svg"))

    end


end # module pandprep
