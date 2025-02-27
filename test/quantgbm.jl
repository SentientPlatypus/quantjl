using Test
using Random
using UnicodePlots
using Plots
using Statistics
include("../quant.jl")
include("../data.jl")


@testset "DDPG" begin

    Random.seed!(3)

    π_ = Net([Layer(101, 80, relu, relu′),
              Layer(80, 64, relu, relu′),
              Layer(64, 32, relu, relu′),
              Layer(32, 16, relu, relu′),
              Layer(16, 1, my_tanh, my_tanh′)], mse_loss, mse_loss′)

    Q̂ = Net([ Layer(102, 80, relu, relu′),  # State + Action as input
              Layer(80, 64, relu, relu′),
              Layer(64, 32, relu, relu′),
              Layer(32, 16, relu, relu′),
              Layer(16, 1, my_tanh, my_tanh′)], mse_loss, mse_loss′)

    γ = 0.95
    τ = 0.1
    quant = Quant(π_, Q̂, γ, τ)

    total_rewards = Float64[]


    capitals = 1000 .+ 200 .* randn(100)
    μ_capital = mean(capitals) # Initial capital in log space
    σ_capital = std(capitals)         # Assume a reasonable standard deviation (can be tuned)

    rewards = randn(100)
    μ_rewards = mean(rewards) 
    σ_rewards = std(rewards)     


    LOOK_BACK_PERIOD = 100
    NUM_EPISODES = 200
    
    price_data = get_historical("SPY")[LOOK_BACK_PERIOD + 1:end] #price percent changes
    price_vscores = get_historical_vscores("SPY", LOOK_BACK_PERIOD) #price vscores
    
    for i in 1:NUM_EPISODES

        println("Episode: $i")

        current_capital = 1000.0
        total_reward = 0;
        d = 0.0
        episode_length = 0


        for t in LOOK_BACK_PERIOD:length(price_vscores) - 1
            if d == 1.0
                break
            end

            episode_length += 1

            # Normalize state
            s = vcat(price_vscores[t - LOOK_BACK_PERIOD + 1:t], [log10(current_capital)])
        
            # Generate action 
            ε = (i <= 100 ) ? randn() * .4 : randn() * 0.2 * exp(-0.002 * i)

            a = clamp(ε + quant.π_(s)[1], -1, 1)
            capital_allocation = current_capital * abs(a)
            current_capital -= capital_allocation

            # Calculate reward
            percent_change = price_data[t + 1]
            raw_r = (a > 0.0 ? 1 : -1) * capital_allocation * (percent_change / 100.0)
            push!(rewards, raw_r)
            μ_rewards = mean(rewards)
            σ_rewards = std(rewards)

            #println("CAPITAL BEFORE: $(current_capital+capital_allocation) ACTION: $a $(a < 0 ? "SHORT" : "LONG"): $capital_allocation PRICE CHANGE: $percent_change REWARD: $raw_r")
            current_capital += raw_r + capital_allocation

            push!(capitals, current_capital)
            μ_capital = mean(capitals)
            σ_capital = std(capitals)


            s′ = vcat(price_vscores[t - LOOK_BACK_PERIOD + 2:t + 1], [log10(current_capital)])

            total_reward += raw_r 
        
            if current_capital < 700.0
                raw_r -= 500
                d = 1.0
            end

            if t == length(price_vscores) - 1
                raw_r += 10 * (current_capital - 1000)
                d = 1.0
            end

            scaled_r = raw_r > 0 ? log10(max(raw_r, 1e-8)) : -log10(max(-raw_r, 1e-8))


            add_experience!(quant, s, a, scaled_r, s′, d)
            train!(quant, 0.0001, 0.0001, 64)
        end
        
        if i % 20 == 0 || i == 1
            # Compute benchmark capital over the same episode length
            benchmark_capital_traj = 1000 * cumprod(1 .+ price_data[LOOK_BACK_PERIOD:LOOK_BACK_PERIOD+episode_length-1] ./ 100)

            # Plot agent's capital trajectory
            Plots.plot(capitals[end - episode_length + 1:end], title="Episode $i Capital over time", 
                    xlabel="Time", ylabel="Capital", label="Agent", lw=2)

            # Overlay benchmark trajectory (Buy & Hold)
            Plots.plot!(benchmark_capital_traj, label="Benchmark (Buy & Hold)", linestyle=:dash, color=:red, lw=2)

            # Save the figure
            Plots.savefig("plots/capital_distribution/higher_gamma_reward_over1000/episode_$(i).png")
        end

        push!(total_rewards, total_reward)
    end 
    
    Plots.histogram(capitals, title="Full Capital Distribution", xlabel="Capital", ylabel="Frequency")
    Plots.savefig("plots/capital_distribution/episodes_full.png")
    plt = Plots.plot(1:NUM_EPISODES, total_rewards, xlabel="Episode", ylabel="total reward", title="DDPG Training")
    Plots.savefig("plots/total_rewards.png")
end