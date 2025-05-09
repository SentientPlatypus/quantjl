using CSV
using DataFrames
include("gbm.jl")


function get_historical(ticker::String)
    #run(`python download.py $ticker`)
    
    df = CSV.read("data/$ticker.csv", DataFrame)
    change_percent = map(row -> Float64(row.changeClosePercent), eachrow(df))
    return reverse(change_percent)
end


function get_historical_raw(ticker::String)
    #run(`python download.py $ticker`)
    
    df = CSV.read("data/$ticker.csv", DataFrame)
    raw = map(row -> Float64(row.close), eachrow(df))
    return reverse(raw)
end

function get_historical_vscores(ticker::String, OBS::Int=100, EPOCH::Int=1000, EXT::Int=20, seed::Int=3)
    raw = get_historical_raw(ticker)
    return vscore(raw, OBS, EPOCH, EXT)
end


function stack_data(tickers::Vector{String}, OBS::Int=100, EPOCH::Int=1000, EXT::Int=20, seed::Int=3)
    data = Dict{String, Vector{Float64}}()
    for ticker in tickers
        data[ticker] = get_historical_vscores(ticker, OBS, EPOCH, EXT, seed)
    end
    return data
end



mutable struct OUNoise
    θ::Float64
    μ::Float64
    σ::Float64
    dt::Float64
    x_prev::Float64
end

function OUNoise(; θ=0.15, μ=0.0, σ=0.2, dt=1.0)
    return OUNoise(θ, μ, σ, dt, 0.0)
end

function sample!(noise::OUNoise)
    x = noise.x_prev + noise.θ * (noise.μ - noise.x_prev) * noise.dt +
        noise.σ * sqrt(noise.dt) * randn()
    noise.x_prev = x
    return x
end

