include("../gbm.jl")
include("../data.jl")
using Test
using Random
using Plots



@testset "GBM NEW" begin
    Random.seed!(3)
    percent_change = get_historical_raw("SPY")
    gbm_path2 = vscore(percent_change)
    plot(gbm_path2[end - 1000:end], title="GBM Path", xlabel="Time", ylabel="Value")
    savefig("plots/gbm_path2.png")

end


@testset "PATH 1000" begin
    Random.seed!(3)
    raw = get_historical_raw("SPY")
    plot(raw, title="SPY_RAW", xlabel="Time", ylabel="Value")
    savefig("plots/appl_1000_raw.png")
end


@testset "PATH_percent 1000" begin
    Random.seed!(3)
    percent_change = get_historical("SPY")
    plot(percent_change, title="SPY_RAW", xlabel="Time", ylabel="Value")
    savefig("plots/appl_1000_pct_change.png")
end


@testset "GBM Hardcoded test" begin
    Random.seed!(3)  # Ensure reproducibility
    LOOK_BACK_PERIOD = 100

    # Get historical percent changes for SPY starting after the lookback period
    percent_change = get_historical("SPY")[LOOK_BACK_PERIOD + 1:end]
    
    # Compute GBM vscores based on the lookback period
    gbm_path2 = get_historical_vscores("SPY", LOOK_BACK_PERIOD)


    gbm_path2 = gbm_path2[end - 6000:end]
    percent_change = percent_change[end - 6000:end]

    # Initialize portfolio parameters
    initial_capital = 1000.0
    capital = initial_capital
    position = 0.0            # Number of shares held (positive for long, negative for short)
    share_price = 100.0       # Assumed starting price of SPY
    capital_over_time = Float64[]  # To track portfolio value over time

    # Track the day index of the last trade.
    # Initialized to -10 so that a trade can occur on the first day (i - (-10) = i+10 >= 10 for i>=0).
    last_trade_day = -10

    for i in 1:length(gbm_path2)
        # Update share price based on the day's percent change.
        share_price *= (1 + percent_change[i] / 100)

        # Compute current portfolio value: cash + (shares held * current share price)
        total_value = capital + position * share_price

        # Only execute a trade if no trade has been made in the past 10 days.
        if i - last_trade_day >= 1
            # Use the ±2 threshold as a trigger, but compute a fraction based on how far past the threshold we are.
            if gbm_path2[i] > 2 && position > 0
                # For scores above 2, compute a fraction to sell.
                # For example, if gbm_path2[i] == 2, fraction is 0; if it reaches 6, fraction is 1 (or capped).
                fraction_to_sell = clamp((gbm_path2[i] - 2) / 4, 0, 1)
                # Use that fraction to determine the sell value. (You can adjust the base fraction if needed.)
                sell_value = fraction_to_sell * total_value  
                sell_shares = sell_value / share_price
                sell_shares = min(sell_shares, position)
                
                capital += sell_shares * share_price
                position -= sell_shares
                last_trade_day = i
            elseif gbm_path2[i] < -2
                # For scores below -2, compute a fraction to invest.
                fraction_to_buy = clamp((-gbm_path2[i] - 2) / 4, 0, 1)
                # Instead of investing all available cash, invest only a fraction.
                invest_cash = fraction_to_buy * capital
                buy_shares = invest_cash / share_price
                
                position += buy_shares
                capital -= invest_cash
                last_trade_day = i
            end
        end
        

        # Recalculate portfolio value after potential trading.
        total_value = capital + position * share_price
        push!(capital_over_time, total_value)
    end

    # Plot the capital trajectory over time.
    plot(capital_over_time, title="Trading with GBM vscore (No Trades in Last 10 Days)",
         xlabel="Time", ylabel="Capital")
    benchmark_capital_traj = 1000 * cumprod(1 .+ percent_change ./ 100)
    plot!(benchmark_capital_traj, label="Benchmark", lw=2)
    savefig("plots/gbm_trading_simulation.png")
    plot(gbm_path2, title="GBM Path vs percent change", xlabel="Time", ylabel="Value")
    plot!(percent_change, label="percent_change", linestyle=:dash, color=:red)
    savefig("plots/gbm_path vs percent_change.png")
end





