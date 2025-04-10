using Test
using Random
using UnicodePlots
using Plots
using Statistics

include("../data.jl")
include("../quant.jl")
include("../data.jl")

function calculate_better_reward(raw_return, capital, prev_capital, risk_window=20, prev_returns=Float64[], risk_aversion=0.1)
    # Track the return
    current_return = raw_return / prev_capital  # Percentage return
    
    # Risk component (using standard deviation if we have enough samples)
    risk_penalty = 0.0
    if length(prev_returns) >= risk_window
        recent_returns = prev_returns[max(1, end-risk_window+1):end]
        risk_penalty = risk_aversion * std(recent_returns)
    end
    
    # Capital protection component - penalize being below initial capital
    capital_penalty = 0.0
    if capital < 1000.0
        capital_penalty = 0.2 * (1000.0 - capital) / 1000.0
    end
    
    # Final reward combines return minus risk and capital penalties
    reward = current_return - risk_penalty - capital_penalty
    
    return reward
end

@testset "DDPG" begin

    Random.seed!(3)

    π_ = Net([Layer(101, 80, relu, relu′),
              Layer(80, 64, relu, relu′),
              Layer(64, 32, relu, relu′),
              Layer(32, 16, relu, relu′),
              Layer(16, 1, idty, idty′)], mse_loss, mse_loss′)

    Q̂ = Net([ Layer(102, 80, relu, relu′),  # State + Action as input
              Layer(80, 64, relu, relu′),
              Layer(64, 32, relu, relu′),
              Layer(32, 16, relu, relu′),
              Layer(16, 1, idty, idty′)], mse_loss, mse_loss′)

    γ = 0.95
    τ = 0.009
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
    
    price_data = get_historical("MSFT")[LOOK_BACK_PERIOD + 1:end] #price percent changes
    price_vscores = get_historical_vscores("MSFT", LOOK_BACK_PERIOD) #price vscores
    ou_noise = OUNoise(θ=0.15, μ=0.0, σ=0.2, dt=1.0) # Initialize OU noise

    recent_returns = Float64[]
    
    for i in 1:NUM_EPISODES

        println("Episode: $i")

        current_capital = 1000.0
        total_reward = 0;
        d = 0.0
        episode_length = 0

        actions = []

        for t in LOOK_BACK_PERIOD:length(price_vscores) - 1
            if d == 1.0
                break
            end

            episode_length += 1

            # Normalize state
            s = vcat(price_vscores[t - LOOK_BACK_PERIOD + 1:t], [log10(current_capital)])
        
            # Generate action 
            ε = sample!(ou_noise)
            a = clamp(quant.π_(s)[1] + ε, -1, 1)
            push!(actions, a)
            ou_noise.σ = max(0.05, ou_noise.σ * exp(-0.0005))

            capital_allocation = current_capital * abs(a)
            prev_capital = current_capital
            current_capital -= capital_allocation

            # Calculate reward
            percent_change = price_data[t + 1]
            raw_r = (a > 0.0 ? 1 : -1) * capital_allocation * (percent_change / 100.0)
            push!(rewards, raw_r)

            push!(recent_returns, raw_r / prev_capital)  # Store return as percentage
            if length(recent_returns) > 100  # Keep a rolling window
                popfirst!(recent_returns)
            end

            #println("CAPITAL BEFORE: $(current_capital+capital_allocation) ACTION: $a $(a < 0 ? "SHORT" : "LONG"): $capital_allocation PRICE CHANGE: $percent_change REWARD: $raw_r")
            current_capital += raw_r + capital_allocation

            push!(capitals, current_capital)
            μ_capital = mean(capitals)
            σ_capital = std(capitals)


            s′ = vcat(price_vscores[t - LOOK_BACK_PERIOD + 2:t + 1], [log10(current_capital)])

            total_reward += raw_r 
        
            if current_capital < 650.0 || t == length(price_vscores) - 1
                d = 1.0
            end

            better_r = calculate_better_reward(raw_r, current_capital, prev_capital, 20, recent_returns)
            add_experience!(quant, s, a, better_r, s′, d)
            train!(quant, 0.0001, 0.0001, 64)
        end
        
        if i % 20 == 0 || i == 1
            # Compute benchmark capital over the same episode length
            benchmark_capital_traj = 1000 * cumprod(1 .+ price_data[LOOK_BACK_PERIOD:LOOK_BACK_PERIOD+episode_length-1] ./ 100)

            # Plot agent's capital trajectory
            capital_plot = plot(capitals[end - episode_length + 1:end], title="Episode $i Capital over time", 
                    xlabel="Time", ylabel="Capital", label="Agent", lw=1)

            # Overlay benchmark trajectory (Buy & Hold)
            plot!(capital_plot, benchmark_capital_traj, label="Benchmark (Buy & Hold)", linestyle=:dash, color=:red, lw=1)

            action_plot = plot(actions, title="Actions over time", xlabel="Time", ylabel="Action", label="Actions", lw=1)

            final_plot = plot(capital_plot, action_plot, layout=(2,1), size=(800,600))
            # Save the figure
            Plots.savefig("plots/capital_distribution/high_frequency_nopos/episode_$(i).png")
        end

        push!(total_rewards, total_reward)
    end 
    
    Plots.histogram(capitals, title="Full Capital Distribution", xlabel="Capital", ylabel="Frequency")
    Plots.savefig("plots/capital_distribution/episodes_full.png")
    plt = Plots.plot(1:NUM_EPISODES, total_rewards, xlabel="Episode", ylabel="total reward", title="DDPG Training")
    Plots.savefig("plots/total_rewards.png")
end