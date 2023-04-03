using DifferentialEquations
using VegaLite
using DataFrames

function sird_ode!(du, u, p, t)
    S, I, R, D = u
    i, r, d = p
    du[1] = -i * S * I
    du[2] = i * S * I - (r + d) * I
    du[3] = r * I
    du[4] = d * I
end

function run_sird(;S=0.99, I=0.01, R=0, D=0, i=0.1, r=0.05, d=0.001, end_time=500)
    initial_conditions=[S, I, R, D]
    parameters=[i, r, d]
    time_span=(0.0, end_time)
    sird_problem = ODEProblem(sird_ode!, initial_conditions, time_span, parameters)
    sird_solution = solve(sird_problem, saveat = 0.1) |> DataFrame
    sird_solution = rename(sird_solution, ["t", "S", "I", "R", "D"])
    return sird_solution
end

function number_of_deaths(;kwargs...)
    simulation_result = run_sird(kwargs)
    number_of_deaths_at_the_end = simulation_result[end, "D"]
    return number_of_deaths_at_the_end
end

run_sird() |>
@vlplot(
    :line,
    transform=[{"fold"=["S", "I", "R", "D"], as=["compartiment", "population"]}],
    x="t:q",
    y="population:q",
    color="compartiment:n"
)
