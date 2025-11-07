using CSV
using DataFrames
using Statistics
using Dates
include("gbm.jl")


function get_historical(ticker::String)
    #run(`python download.py $ticker`)
    
    df = CSV.read("data/$ticker.csv", DataFrame)
    change_percent = map(row -> Float64(row.changeClosePercent), eachrow(df))
    return reverse(change_percent)
end


function get_historical_raw_list(ticker::String)
    #run(`python download.py $ticker`)
    
    df = CSV.read("data/$ticker.csv", DataFrame)
    raw = map(row -> Float64(row.close), eachrow(df))
    return reverse(raw)
end

function get_historical_raw(ticker::String)
    df = CSV.read("data/$(ticker).csv", DataFrame)
    sort!(df, :date)
    return df
end


# 0. VSCORES.
function get_historical_vscores(ticker::String, OBS::Int=100, EPOCH::Int=1000, EXT::Int=20, seed::Int=3)
    raw = get_historical_raw_list(ticker)
    return vscore(raw, OBS, EPOCH, EXT)
end


# 2. Exponential Moving Average (EMA)
function ema_series(ticker::String, window::Int=14)
    df = get_historical_raw(ticker)
    
    n = nrow(df)

    if window <= 0 || n < window
        throw(ArgumentError("Window size must be > 0 and <= length of price series"))
    end

    α = 2.0 / (window + 1)
    ema = Vector{Union{Missing, Float64}}(undef, n)

    # First EMA value is just a simple average
    ema[1:window-1] .= NaN
    ema[window] = mean(df.close[1:window])

    for t in (window+1):n
        ema[t] = α * df.close[t] + (1 - α) * ema[t-1]
    end

    return ema
end

# 3. RSI
function rsi_series(ticker::String, window::Int=14)
    df = get_historical_raw(ticker)
    
    n = nrow(df)

    if window <= 0 || n < window 
        throw(ArgumentError9("Window size must be greater than zero and larger than the length of the price series"))
    end

    deltas = diff(df.close)
    gains = max.(deltas, 0.0)
    losses = -min.(deltas, 0.0)

    avg_gain = Vector{Float64}(undef, n-1)
    avg_loss = Vector{Float64}(undef, n-1)

    # Initial averages
    avg_gain[1:window-1] .= NaN
    avg_loss[1:window-1] .= NaN
    avg_gain[window] = mean(gains[1:window])
    avg_loss[window] = mean(losses[1:window])

    # Wilder's smoothing
    for i in (window+1):(n-1)
        avg_gain[i] = (avg_gain[i-1] * (window - 1) + gains[i]) / window
        avg_loss[i] = (avg_loss[i-1] * (window - 1) + losses[i]) / window
    end

    # RSI computation
    rsi = Vector{Float64}(undef, n)
    rsi[1:window] .= NaN
    for i in (window+1):n
        rs = avg_loss[i-1] == 0 ? Inf : avg_gain[i-1] / avg_loss[i-1]
        rsi[i] = 100.0 - (100.0 / (1 + rs))
    end

    return rsi
end

# 5. Bollinger Band %B
function bb_percentb_series_safe(ticker::String, window::Int=20)
    df = get_historical_raw(ticker)
    n  = nrow(df)
    ma = [i < window ? NaN : mean(@view df.close[i-window+1:i]) for i in 1:n]
    sd = [i < window ? NaN : std(@view df.close[i-window+1:i])  for i in 1:n]

    upper = ma .+ 2 .* sd
    lower = ma .- 2 .* sd
    bandw = upper .- lower

    percentb = similar(df.close, Float64)
    @inbounds for i in 1:n
        if isnan(bandw[i]) || bandw[i] ≈ 0.0
            # If the band collapses, price == ma; set %B to 0.5 (center of band)
            percentb[i] = 0.5
        else
            percentb[i] = (df.close[i] - lower[i]) / bandw[i]
        end
    end
    return percentb
end
# 7. VWAP
function vwap_series(ticker::String)
    df = get_historical_raw(ticker)
    typical_price = (df.high .+ df.low .+ df.close) ./ 3
    cum_vp = cumsum(typical_price .* df.volume)
    cum_vol = cumsum(df.volume)
    return cum_vp ./ cum_vol
end

# 9. Time-of-Day Feature (sin/cos of minutes since open)
function time_of_day_features(ticker::String)
    df = get_historical_raw(ticker)
    # Parse each date string to DateTime using the correct format
    time = [DateTime(dt, dateformat"yyyy-mm-dd HH:MM:SS") for dt in df.date]
    minutes = [Dates.hour(t)*60 + Dates.minute(t) for t in time]
    minutes_in_day = 7*60
    sin_feat = sin.(2π .* minutes ./ minutes_in_day)
    cos_feat = cos.(2π .* minutes ./ minutes_in_day)
    return sin_feat, cos_feat
end

function macd_series(ticker::String; short_period::Int = 12, long_period::Int = 26, signal_period::Int = 9)
    df = get_historical_raw(ticker)
    close = df.close

    # EMA helper function
    function ema(series::Vector{Float64}, period::Int)
        alpha = 2.0 / (period + 1)
        ema_series = similar(series)
        ema_series[1] = series[1]  # Initialize with first value
        for i in 2:length(series)
            ema_series[i] = alpha * series[i] + (1 - alpha) * ema_series[i-1]
        end
        return ema_series
    end

    short_ema = ema(close, short_period)
    long_ema = ema(close, long_period)
    macd_line = short_ema .- long_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line .- signal_line

    return histogram
end

function volume_series(ticker::String)
    df = get_historical_raw(ticker)
    log10_volume = log.(1000, df.volume .+ 1)
    return log10_volume
end

# Combine all features
function get_all_features(ticker::String, day::Int, LOOK_BACK_PERIOD::Int=100)
    df = DataFrame()

    current_date_str = Dates.format(Dates.today(), "yyyy-mm-dd")

    filepath_name = "$(current_date_str)/$(ticker)_day$(day)"

    df.vscores = get_historical_vscores(filepath_name, LOOK_BACK_PERIOD)

    df.ema = ema_series(filepath_name)[LOOK_BACK_PERIOD+1:end]  
    # df.volume = volume_series(filepath_name)[LOOK_BACK_PERIOD+1:end]
    df.rsi = rsi_series(filepath_name)[LOOK_BACK_PERIOD+1:end]  
    df.macd = macd_series(filepath_name)[LOOK_BACK_PERIOD+1:end] 
    df.bb_percentb = bb_percentb_series_safe(filepath_name)[LOOK_BACK_PERIOD+1:end]
    # df.vwap = vwap_series(filepath_name)[LOOK_BACK_PERIOD+1:end] 
    sin_feat, cos_feat = time_of_day_features(filepath_name)

    df.sin_feat = sin_feat[LOOK_BACK_PERIOD+1:end]
    df.cos_feat = cos_feat[LOOK_BACK_PERIOD+1:end]
    
    df_standardized = deepcopy(df)
    cols_to_standardize = [:rsi, :ema]
    for col in cols_to_standardize
        μ = mean(df[!, col])
        σ = std(df[!, col])
        σ = max(σ, 1e-8)  # epsilon floor
        df_standardized[!, col] = (df[!, col] .- μ) ./ σ
    end

    day_price = get_historical_raw(filepath_name).changeClosePercent[LOOK_BACK_PERIOD+1 : end]
    return df_standardized, day_price
end

function get_month_features(ticker::String, days::Int=30, LOOK_BACK_PERIOD=100)
    dataframes = []
    prices = []

    for day in 1:days

        try
            df, day_price = get_all_features(ticker, day, LOOK_BACK_PERIOD)

            push!(prices, day_price)
            push!(dataframes, df)
        catch e
            @warn "Failed to get features for day $day: $(e)"
            continue
        end
    end
    return dataframes, prices
end


##NOISE_------------------------------------------
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

