using CSV
using DataFrames
# Read the CSV data
df = CSV.read("data/AAPL.csv", DataFrame)

# Extract the `changePercent` column as a Vector{Float64}
change_percent = map(row -> Float64(row.changePercent), eachrow(df))




function get_historical(ticker::String)
    #run(`python download.py $ticker`)
    
    df = CSV.read("data/$ticker.csv", DataFrame)
    change_percent = map(row -> Float64(row.changePercent), eachrow(df))
    return reverse(change_percent)
end


function get_historical_raw(ticker::String)
    #run(`python download.py $ticker`)
    
    df = CSV.read("data/$ticker.csv", DataFrame)
    change_percent = map(row -> Float64(row.close), eachrow(df))
    return reverse(change_percent)
end



function get_historical_vscores(ticker::String, OBS::Int=60, EPOCH::Int=1000, EXT::Int=20, seed::Int=3)
    raw = get_historical_raw(ticker)
    return vscore(raw, OBS, EPOCH, EXT, seed)
end


