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

    γ = 0.5
    τ = 0.01
    quant = Quant(π_, Q̂, γ, τ)

    price_data = get_historical("AAPL") #price percent changes
    mean_price = mean(price_data)
    std_price = std(price_data)
    normalized_price_data = (price_data .- mean_price) ./ std_price
    
    total_rewards = Float64[]
    
    mean_capital = log10(1000.0)  # Initial capital in log space
    std_capital = 1.0            # Assume a reasonable standard deviation (can be tuned)
    
    LOOK_BACK_PERIOD = 100
    NUM_EPISODES = 1000
    
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

            episode_length += 1
            if episode_length > 1000 || current_capital < 100.0
                d = 1.0
                break
            end

            # Normalize state
            s = vcat(normalized_price_data[t - LOOK_BACK_PERIOD + 1:t], [(log10(current_capital) - mean_capital) / std_capital])
            s′ = vcat(normalized_price_data[t - LOOK_BACK_PERIOD + 2:t + 1], [(log10(current_capital) - mean_capital) / std_capital])
        
            # Generate action and scale reward
            a = clamp(randn() + quant.π_(s)[1], -1, 1)
            capital_allocation = min(current_capital * abs(a), .5)
            current_capital -= capital_allocation
            percent_change = price_data[t + 1]
    
            r = (a > 0.0 ? 1 : -1) * capital_allocation * (percent_change / 100.0)
            max_reward = max(max_reward, r)  # Update max_reward
            scaled_r = r / max_reward  # Scale reward
        
            # println("CAPITAL BEFORE: $(current_capital+capital_allocation) ACTION: $a $(a < 0 ? "SHORT" : "LONG"): $capital_allocation PRICE CHANGE: $percent_change REWARD: $scaled_r")
            # Update capital
            current_capital += scaled_r * max_reward + capital_allocation


            total_reward += scaled_r
        
            # Train
            add_experience!(quant, s, a, scaled_r, s′, d)
            train!(quant, 0.00001, 0.0, 64)
        end
        
        push!(total_rewards, total_reward)
    end 
    
    plt = UnicodePlots.lineplot(1:NUM_EPISODES, log10.(total_rewards), xlabel="Episode", ylabel="Highest Capital (log)", title="DDPG Training", width=100)
    display(plt)
end