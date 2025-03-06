include("../gbm.jl")
include("../data.jl")
using Test
using Random
using Plots



@testset "GBM NEW" begin
    Random.seed!(3)
    percent_change = get_historical_raw("AAPL")
    gbm_path2 = vscore(percent_change)
    plot(gbm_path2[end - 360:end], title="GBM Path", xlabel="Time", ylabel="Value")
    savefig("plots/gbm_path2.png")

end


@testset "PATH 1000" begin
    Random.seed!(3)
    raw = get_historical_raw("AAPL")
    plot(raw, title="AAPL_RAW", xlabel="Time", ylabel="Value")
    savefig("plots/appl_1000_raw.png")
end


@testset "PATH_percent 1000" begin
    Random.seed!(3)
    percent_change = get_historical("AAPL")
    plot(percent_change, title="AAPL_RAW", xlabel="Time", ylabel="Value")
    savefig("plots/appl_1000_pct_change.png")
end


@testset "GBM Hardcoded Test" begin
    using Random, Plots

    # Ensure reproducibility
    Random.seed!(3)

    # Constants
    LOOK_BACK_PERIOD = 60
    GRAPH_LENGTH = 1000
    INITIAL_CAPITAL = 1000.0
    TRADE_COOLDOWN = 5  # Days between trades
    SELL_THRESHOLD = 2
    BUY_THRESHOLD = -2

    # Load historical data
    percent_change = get_historical("AAPL")[LOOK_BACK_PERIOD + 1:end]
    real_price = get_historical_raw("AAPL")[LOOK_BACK_PERIOD + 1:end]
    gbm_scores = get_historical_vscores("AAPL", LOOK_BACK_PERIOD)

    # Trim data to the desired length
    real_price = real_price[end - GRAPH_LENGTH:end]
    gbm_scores = gbm_scores[end - GRAPH_LENGTH:end]
    percent_change = percent_change[end - GRAPH_LENGTH:end]

    # Initialize portfolio tracking
    capital = INITIAL_CAPITAL
    position = 0.0  # Number of shares held
    share_price = 100.0  # Assumed starting price
    portfolio_value = Float64[]
    capital_over_time = Float64[]
    last_trade_day = -10  # Allow trading on the first day

    # Trading simulation
    for i in 1:length(gbm_scores)
        # Update share price based on percent change
        share_price *= (1 + percent_change[i] / 100)
        total_value = capital + position * share_price


        # Trading logic (ensure cooldown period)
        if i - last_trade_day >= TRADE_COOLDOWN
            if gbm_scores[i] > SELL_THRESHOLD && position > 0
                
                fraction_to_sell = 0.4
                sell_value = fraction_to_sell * total_value  
                sell_shares = min(sell_value / share_price, position)

                capital += sell_shares * share_price
                position -= sell_shares
                last_trade_day = i
                println("Capital: $capital signal at day $i: gbm_scores[$i] = $(gbm_scores[i])")
            elseif gbm_scores[i] < BUY_THRESHOLD
                fraction_to_buy = 1  # Can be adjusted dynamically
                invest_cash = fraction_to_buy * capital
                buy_shares = invest_cash / share_price

                position += buy_shares
                capital -= invest_cash
                last_trade_day = i
            end
        end

        # Track portfolio value
        push!(capital_over_time, capital)
        push!(portfolio_value, capital + position * share_price)
    end

    # Generate plots
    p1 = plot(portfolio_value, title="Trading with GBM vscore", xlabel="Time", ylabel="Capital")
    benchmark_capital = INITIAL_CAPITAL * cumprod(1 .+ percent_change ./ 100)
    plot!(benchmark_capital, label="Benchmark", lw=2)

    p2 = plot(gbm_scores, title="GBM Path", xlabel="Time", ylabel="Value")
    p3 = plot(real_price, title="Real Price", xlabel="Time", ylabel="Value", label="Real Price", color=:red)
    p4 = plot(percent_change, title="Percent Change", xlabel="Time", ylabel="Value", label="Percent Change", color=:blue)
    p5 = plot(capital_over_time, title="Capital Over Time", xlabel="Time", ylabel="Value", label="Capital Over Time", color=:green)

    # Save plots
    plot(p1, p2, layout=(2,1), size=(800, 600), legend=:topright)
    savefig("plots/gbm_trading_simulation.png")

    plot(p2, p3, p4, p5, layout=(4,1), size=(800, 900))
    savefig("plots/gbm_analysis.png")
end





