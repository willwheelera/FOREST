using JuMP
using BARON
using Parquet2, DataFrames
#ENV["BARON_EXEC"] = "/home/wwheele1/baron-lin64/baron"

Nmax = 5 # replacements per year
C_U = 5. # cost to upgrade (assume units of thousand$)
C_D = 0.1 # value to defer each year
C_F = 6. # cost of failure (including replacement with upgrade)
ntf = 50 # transformers to consider in optimization

filepath = "../output/alburgh_tf_failure_curves_MH_2025_20years.parquet"
df = Parquet2.Dataset(filepath) |> DataFrame
select!(df, Not(:__index_level_0__))
println(size(df))

last_row_values = collect(df[end, :])
top_indices = partialsortperm(last_row_values, 1:ntf, rev=true)
print(top_indices)
df = df[collect(8759:8760:end), top_indices]
#df = df[1:10, :]
nyear, ntf = size(df)

F = Matrix(df)

model = Model(BARON.Optimizer)
set_attribute(model, "PrLevel", 1)

@variable(model, U[1:nyear,1:ntf], Bin)
@objective(model, Min, 
    sum(U)*C_U - sum(y*sum(U[y, :]) for y in 1:nyear) * C_D + 
    sum(U.*F) * C_F + ((1. .-sum(U, dims=1)) * F[end, :])[1] * C_F
)
@constraint(model, [x in 1:ntf], sum(U[:, x]) <= 1)
@constraint(model, [y in 1:nyear], sum(U[y, :]) <= Nmax)

#print(model)
optimize!(model)

u = value.(U)
upgrades = [findfirst(Bool.(u[:, x])) for x in 1:ntf]
upgrades = map(x-> isnothing(x) ? Int(0) : x, upgrades)
N_failures = sum(u.*F) + ((1. .-sum(u, dims=1)) * F[end, :])[1]
println(N_failures)
println(round.(F[end, :], digits=2))
println(upgrades)

out = DataFrame(tf=>names(df), year=>upgrades)
out = out[out.year>0, :]
Parquet2.writefile("opt_MH", out)
