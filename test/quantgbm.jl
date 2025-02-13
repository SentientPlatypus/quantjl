using Test
using Random
using UnicodePlots
using Statistics
include("../quant.jl")
include("../data.jl")


@testset "DDPG" begin

    Random.seed!(3)

    π_ = Net([Layer(101, 64, relu, relu′),
              Layer(64, 32, relu, relu′),
              Layer(32, 16, relu, relu′),
              Layer(16, 1, my_tanh, my_tanh′)], mse_loss, mse_loss′)

    Q̂ = Net([Layer(102, 64, relu, relu′),  # State + Action as input
              Layer(64, 32, relu, relu′),
              Layer(32, 16, relu, relu′),
              Layer(16, 1, my_tanh, my_tanh′)], mse_loss, mse_loss′)

    γ = 0.9
    τ = 0.09
    quant = Quant(π_, Q̂, γ, τ)

    price_data = get_historical("AAPL") #price percent changes
    price_vscores = get_historical_vscores("AAPL") #price vscores
    

    total_rewards = Float64[]

    capitals = Float64[]
    mean_capital = log10(1000.0)  # Initial capital in log space
    std_capital = 1.0            # Assume a reasonable standard deviation (can be tuned)


    LOOK_BACK_PERIOD = 100
    NUM_EPISODES = 100
    
    for i in 1:NUM_EPISODES
        if (i % 10 == 0)
            println("Episode: $i")
        end
        current_capital = 1000.0
        total_reward = 0;
        max_reward = -Inf  # Initialize max_reward for the episode
      
        d = 0.0
        episode_length = 0
    
        for t in LOOK_BACK_PERIOD:length(price_data) - 1
            if d == 1.0
                break
            end

            episode_length += 1

            # Normalize state
            s = vcat(price_vscores[t - LOOK_BACK_PERIOD + 1:t], [(log10(current_capital) - mean_capital) / std_capital])
        
            # Generate action and scale reward
            ε = randn() * .25
            a = clamp(ε + quant.π_(s)[1], -1, 1)
            capital_allocation = current_capital * min(abs(a), .5)
            current_capital -= capital_allocation

            percent_change = price_data[t + 1]
            r = (a > 0.0 ? 1 : -1) * capital_allocation * (percent_change / 100.0)
            max_reward = max(max_reward, r)  # Update max_reward
            scaled_r = r / max_reward  # Scale reward

            # println("CAPITAL BEFORE: $(current_capital+capital_allocation) ACTION: $a $(a < 0 ? "SHORT" : "LONG"): $capital_allocation PRICE CHANGE: $percent_change REWARD: $scaled_r")
        
            current_capital += scaled_r * max_reward + capital_allocation

            push!(capitals, current_capital)
            if length(capitals) > 100
                mean_capital = mean(log10.(capitals))
                std_capital = std(log10.(capitals))
            end

            s′ = vcat(price_vscores[t - LOOK_BACK_PERIOD + 2:t + 1], [(log10(current_capital) - mean_capital) / std_capital])

            total_reward += scaled_r * max_reward
        
            if episode_length > 1000 || current_capital < 100.0
                r -= 500  # Harsh penalty for depleting capital
                scaled_r = r / max_reward
                d = 1.0
            end

            # Train 
            add_experience!(quant, s, a, scaled_r, s′, d)
            train!(quant, 0.0001, 0.0001, 64)
        end
        
        push!(total_rewards, total_reward)
    end 
    
    plt = UnicodePlots.lineplot(1:NUM_EPISODES, total_rewards, xlabel="Episode", ylabel="total reward", title="DDPG Training", width=100)
    display(plt)
end