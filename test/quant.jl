using Test
using Random
using UnicodePlots
include("../quant.jl")
include("../data.jl")


@testset "DDPG" begin
    π_ = Net([Layer(101, 64, relu, relu′),
              Layer(64, 64, relu, relu′),
              Layer(64, 1, (x) -> tanh.(x), (x) -> 1 .- x.^2)], mse_loss, mse_loss′)

    Q̂ = Net([Layer(102, 64, relu, relu′),  # State + Action as input
              Layer(64, 64, relu, relu′),
              Layer(64, 1, relu, relu′)], mse_loss, mse_loss′)

    γ = 0.99
    τ = 0.01
    quant = Quant(π_, Q̂, γ, τ)

    price_data = get_historical("AAPL") #price percent changes

    LOOK_BACK_PERIOD = 100
    NUM_EPISODES = 1000

    for i in 1:NUM_EPISODES
        current_capital = 1000.0
        for t in LOOK_BACK_PERIOD:length(price_data)
            println("Current capital: $current_capital")
            if current_capital <= 0  #Terminal state.
                break
            end

            s = vcat(price_data[t - LOOK_BACK_PERIOD + 1:t], [current_capital])
            a = clamp(randn() + quant.π_(s)[1], -1, 1) # a = clip(π(s) + ϵ, -1, 1)

            capital_allocation = current_capital * abs(a)

            # Reward (use price difference to compute profit/loss)
            price_change = price_data[t + 1] - price_data[t]
            position_value = a * capital_allocation
            r = position_value * price_change / price_data[t]  # Realized P&L

            # Update capital
            current_capital += r

            # Next state
            s′ = vcat(price_data[t - LOOK_BACK_PERIOD + 2:t + 1], [current_capital])

            # Store experience and train
            add_experience!(quant, s, a, r, s′)
            train!(quant, 0.001, 0.01, 64)
 
        end
    end 
end