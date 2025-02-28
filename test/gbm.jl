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


    gbm_path2 = gbm_path2[end - 100:end]
    percent_change = percent_change[end - 100:end]

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
        if i - last_trade_day >= 10
            if gbm_path2[i] > 2 && position > 0
                # If vscore > 2 and we hold a long position, sell 30% of the portfolio's value.
                sell_value = 0.4 * total_value        # 30% of the portfolio value
                sell_shares = sell_value / share_price  # Convert value to number of shares
                sell_shares = min(sell_shares, position)  # Do not sell more than we own

                capital += sell_shares * share_price  # Increase cash by sale proceeds
                position -= sell_shares               # Decrease our position
                last_trade_day = i                    # Update last trade day
            elseif gbm_path2[i] < -2
                # If vscore < -2, buy as many shares as possible with available cash.
                buy_shares = capital / share_price
                position += buy_shares  # Increase position by the shares purchased
                capital = 0           # All cash is invested
                last_trade_day = i    # Update last trade day
            end
        end

        # Recalculate portfolio value after potential trading.
        total_value = capital + position * share_price
        push!(capital_over_time, total_value)
    end

    # Plot the capital trajectory over time.
    plot(capital_over_time, title="Trading with GBM vscore (No Trades in Last 10 Days)",
         xlabel="Time", ylabel="Capital")
    savefig("plots/gbm_trading_simulation.png")
    plot(gbm_path2, title="GBM Path vs percent change", xlabel="Time", ylabel="Value")
    plot!(percent_change, label="percent_change", linestyle=:dash, color=:red)
    savefig("plots/gbm_path vs percent_change.png")
end





