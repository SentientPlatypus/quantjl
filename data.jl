using CSV
using DataFrames
# Read the CSV data
df = CSV.read("data/AAPL.csv", DataFrame)

# Extract the `changePercent` column as a Vector{Float64}
change_percent = map(row -> Float64(row.changePercent), eachrow(df))




function get_historical(ticker::String)
    run(`python data/download.py $ticker`)
    
    df = CSV.read("data/$ticker.csv", DataFrame)
    change_percent = map(row -> Float64(row.changePercent), eachrow(df))
    return change_percent
end
