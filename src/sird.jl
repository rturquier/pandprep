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

function run_sird(S=0.99, I=0.01, R=0, D=0, i=0.2, r=0.05, d=0.01, end_time=500)
    initial_conditions=[S, I, R, D]
    parameters=[i, r, d]
    time_span=(0.0, end_time)
    sird_problem = ODEProblem(sird_ode!, initial_conditions, time_span, parameters)
    sird_solution = solve(sird_problem, saveat = 0.1) |> DataFrame
    sird_solution = rename(sird_solution, ["t", "S", "I", "R", "D"])
    return sird_solution
end

function count_final_deaths(sird_result)
    return sird_result[end, "D"]
end

run_sird() |>
@vlplot(
    :line,
    transform=[{fold=["S", "I", "R", "D"], as=["compartiment", "population"]}],
    x="t:q",
    y="population:q",
    color="compartiment:n"
)


final_deaths_df = DataFrame(S = 0:50:100000)
final_deaths_df = transform(
    final_deaths_df,
    :S => (x -> count_final_deaths.(run_sird.(x))) => :deaths
)

final_deaths_df |>
@vlplot(:line, :S, :deaths)
