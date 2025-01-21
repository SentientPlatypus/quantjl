using Test
using Random
using UnicodePlots
include("../quant.jl")
include("../data.jl")


@testset "DDPG" begin

    Random.seed!(3)

    π_ = Net([Layer(101, 64, relu, relu′),
              Layer(64, 32, relu, relu′),
              Layer(32, 16, relu, relu′),
              Layer(16, 1, (x) -> tanh.(x), (x) -> 1 .- x.^2)], mse_loss, mse_loss′)

    Q̂ = Net([Layer(102, 64, relu, relu′),  # State + Action as input
              Layer(64, 32, relu, relu′),
              Layer(32, 16, relu, relu′),
              Layer(16, 1, relu, relu′)], mse_loss, mse_loss′)

    γ = 0.5
    τ = 0.01
    quant = Quant(π_, Q̂, γ, τ)

    price_data = get_historical("AAPL") #price percent changes

    highest_capitals = Float64[]

    LOOK_BACK_PERIOD = 100
    NUM_EPISODES = 1000

    for i in 1:NUM_EPISODES
        if (i % 1000 == 0)
            println("Episode: $i")
        end
        current_capital = 1000.0
        max_capital = current_capital

        d = 0.0
        for t in LOOK_BACK_PERIOD:length(price_data)

            if d == 1.0
                break
            end

            if current_capital <= 0  #Terminal state.
                d = 1.0
            end

            s = vcat(price_data[t - LOOK_BACK_PERIOD + 1:t], [current_capital])
            a = clamp(randn() + quant.π_(s)[1], -1, 1) # a = clip(π(s) + ϵ, -1, 1)


            capital_allocation = current_capital * min(abs(a), .1) #limit allocation to ten percent of capital.
            price_change = price_data[t + 1] - price_data[t]
            position_value = a * capital_allocation
            r = position_value * price_change / price_data[t]  # Realized P&L

            # Update capital
            current_capital += r
            max_capital = max(max_capital, current_capital)

            # Next state
            s′ = vcat(price_data[t - LOOK_BACK_PERIOD + 2:t + 1], [current_capital])

            # Store experience and train
            add_experience!(quant, s, a, r, s′, d)
            train!(quant, 0.001, 0.01, 64)
        end

        push!(highest_capitals, max_capital)
    end 


    plt = UnicodePlots.lineplot(1:NUM_EPISODES, log10.(highest_capitals), xlabel="Episode", ylabel="Highest Capital (log)", title="DDPG Training", width=100)
    display(plt)
end