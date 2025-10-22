using JuMP
using BARON
using Parquet2, DataFrames
#ENV["BARON_EXEC"] = "/home/wwheele1/baron-lin64/baron"

for Nmax in [5, 7, 10, 20] # replacements per year
C_U = 5. # cost to upgrade (assume units of thousand$)
C_D = 0.1 # value to defer each year
C_F = 10. # cost of failure (including replacement with upgrade)
C_EF = 1. # quadratic cost of simultaneous failure
ntf = 250 # transformers to consider in optimization

GROWTH = "MH"
filepath = "../output/alburgh_tf_failure_curves_$(GROWTH)_2025_20years_sampled.parquet"
df = Parquet2.Dataset(filepath) |> DataFrame
select!(df, Not(:__index_level_0__))
println(size(df))

last_row_values = collect(df[end, :])
top_indices = partialsortperm(last_row_values, 1:ntf, rev=true)
#println(top_indices)
#df = df[collect(8759:8760:end), top_indices] # load pre-sampled data
df = df[:, top_indices]
#df = df[1:10, :]
nyear, ntf = size(df)
println(size(df))

F = Matrix(df)

model = Model(BARON.Optimizer)
set_attribute(model, "PrLevel", 1)

@variable(model, U[1:nyear,1:ntf], Bin)
@variable(model, Efail[1:nyear])
@objective(model, Min, 
    sum(U)*C_U - sum(y*sum(U[y, :]) for y in 1:nyear) * C_D + 
    sum(U.*F) * (C_F-C_U) + ((1. .-sum(U, dims=1)) * F[end, :])[1] * C_F
    + sum(Efail.^2) * C_EF
)
@constraint(model, [x in 1:ntf], sum(U[:, x]) <= 1)
@constraint(model, [y in 1:nyear], sum(U[y, :]) <= Nmax)
@constraint(model, [y in 2:nyear], Efail[y] == sum((1 .-sum(U[1:y, :], dims=1)[1,:]) .* (F[y, :] .- F[y-1, :])))

optimize!(model)

u = value.(U)
upgrades = [findfirst(u[:, x] .> 0.5) for x in 1:ntf]
upgrades = map(x-> isnothing(x) ? Int(0) : x, upgrades)
N_failures = sum(u.*F) + ((1. .-sum(u, dims=1)) * F[end, :])[1]
println(N_failures)
println(round.(F[end, :], digits=2))
println(upgrades)
cost = objective_value(model)
println(cost)
println()
println("Nmax \t", Nmax)
println("U  \t", C_U, " \t", sum(u))
println("D  \t", C_D, " \t", sum(y*sum(u[y, :]) for y in 1:nyear))
println("F1 \t", C_F, " \t", sum(u.*F))
println("F2 \t", C_F, " \t", ((1. .-sum(u, dims=1)) * F[end, :])[1])
Ef = value.(Efail)
println("EF \t", C_EF, " \t", sum(Ef.^2))
println("cost  ", 
    sum(u)*C_U - sum(y*sum(u[y, :]) for y in 1:nyear) * C_D + 
    sum(u.*F) * (C_F-C_U) + ((1. .-sum(u, dims=1)) * F[end, :])[1] * C_F
    + sum(Ef.^2) * C_EF
)

out = DataFrame(:tf=>names(df), :year=>upgrades)
out = out[out.year.>0, :]
Parquet2.writefile("opt_$(GROWTH)_capF_$(Nmax).parquet", out)

end
