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

parms = [0.1, 0.05, 0.001]
init = [0.99, 0.01, 0.0, 0.0]
tspan = (0.0, 500.0)
sird_problem = ODEProblem(sird_ode!, init, tspan, parms)
sird_solution = solve(sird_problem, saveat = 0.1) |> DataFrame
sird_solution = rename(sird_solution, ["t", "S", "I", "R", "D"])

sird_solution |>
@vlplot(
    :line,
    transform=[{"fold"=["S", "I", "R", "D"], as=["compartiment", "population"]}],
    x="t:q",
    y="population:q",
    color="compartiment:n"
)

sird_solution[end, "D"]
