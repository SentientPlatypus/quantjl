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

    ticker = "MSFT"
    LOOK_BACK_PERIOD = 30
    NUM_EPISODES = 250

    month_features, month_prices = get_month_features(ticker, 30,LOOK_BACK_PERIOD)
    nIndicators = ncol(month_features[1])
    
    π_ = Net([
        Layer(nIndicators * LOOK_BACK_PERIOD + 1, 200, relu, relu′),
        Layer(200, 100, relu, relu′),
        Layer(100, 50, relu, relu′),
        Layer(50, 30, relu, relu′),
        Layer(30, 10, relu, relu′),
        Layer(10, 1, sigmoid, sigmoid′)
    ], mse_loss, mse_loss′)

    Q̂ = Net([
        Layer(nIndicators * LOOK_BACK_PERIOD  + 2, 200, relu, relu′),
        Layer(200, 100, relu, relu′),
        Layer(100, 50, relu, relu′),
        Layer(50, 30, relu, relu′),
        Layer(30, 10, relu, relu′),
        Layer(10, 1, idty, idty′)
    ], mse_loss, mse_loss′)


    println("STARTING TRAINING FOR $(ticker). Using features: $(names(month_features[1]))")

    γ = 0.99
    τ = 0.005
    quant = Quant(π_, Q̂, γ, τ)

    total_rewards = Float64[]


    rewards = randn(100) 
    ou_noise = OUNoise(θ=0.15, μ=0.0, σ=0.2, dt=1.0) # Initialize OU noise

    
    
    for i in 1:NUM_EPISODES

        println("Episode: $i")

        current_capital = 1000.0
        episode_rewards = Float64[]
        d = 0.0
        episode_length = 0
        
        # Track current allocation
        current_market_allocation = 0.0  # Start with 0% allocated (all cash)
        actions = []
        capitals = [current_capital]      # <-- reset here
        recent_returns = Float64[]
        

        day = rand(1:28)

        day_features = month_features[day]
        day_change = month_prices[day]
        
        for t in (LOOK_BACK_PERIOD):nrow(day_features) - 1
            if d == 1.0
                break
            end
    
            episode_length += 1
    
            # Normalize state
            s = vcat([day_features[!, col][t - LOOK_BACK_PERIOD + 1:t] for col in names(day_features)]..., [log10(current_capital)])        
            # Generate action (target allocation)
            ε = sample!(ou_noise)
            target_allocation = clamp(quant.π_(s)[1] + ε, 0, 1)

            push!(actions, quant.π_(s)[1]) 
            ou_noise.σ = max(0.05, ou_noise.σ * exp(-0.00005))
    
            # Calculate change in allocation
            allocation_change = target_allocation - current_market_allocation
            
            # Apply market impact/transaction costs (optional)
            transaction_cost = 0.0002 * abs(allocation_change) * current_capital
            current_capital -= transaction_cost
            
            # Record capital before market moves
            prev_capital = current_capital
            
            # Apply market movement to existing allocation
            current_market_allocation = target_allocation

            percent_change = day_change[t]
            market_return = current_market_allocation * current_capital * (percent_change / 100.0)
            current_capital += market_return
            

            # Calculate reward
            raw_r = market_return - transaction_cost
            push!(rewards, raw_r)
            push!(recent_returns, raw_r / prev_capital)  # Store return as percentage
            if length(recent_returns) > 100
                popfirst!(recent_returns)
            end
            better_r = calculate_better_reward(raw_r, current_capital, prev_capital, 20, recent_returns)
            if better_r < 0.0
                better_r *= 3
            end

            push!(capitals, current_capital)
    
            s′ = vcat([day_features[!, col][t - LOOK_BACK_PERIOD + 2:t+1] for col in names(day_features)]..., [log10(current_capital)])
    
            push!(episode_rewards, raw_r)
        
            if current_capital < 950.0 || t == nrow(day_features) - 1
                d = 1.0
                extra_reward = 10 * (current_capital - 1000.0) / 1000.0
                better_r += extra_reward
            end
    
            
            
            add_experience!(quant, s, target_allocation, better_r, s′, d)
            train!(quant, 3e-4, 1e-4, 0.0001, 256)
        end
        
        if i % 20 == 0 || i == 1
            # Compute benchmark capital over the same episode length
            benchmark_capital_traj = 1000 * cumprod(1 .+ day_change[LOOK_BACK_PERIOD:LOOK_BACK_PERIOD + episode_length] ./ 100)

            # Plot agent's capital trajectory
            capital_plot = plot(capitals, title="Episode $i Capital over time", 
                xlabel="Time", ylabel="Capital", label="Agent", lw=1)

            # Overlay benchmark trajectory (Buy & Hold)
            plot!(capital_plot, benchmark_capital_traj, label="Benchmark (Buy & Hold)", linestyle=:dash, color=:red, lw=1)
            
            action_plot = plot(actions, title="Actions over time", xlabel="Time", ylabel="Action", label="Actions", lw=1)

            final_plot = plot(capital_plot, action_plot, layout=(2,1), size=(800,600))
            # Save the figure

            date_str = Dates.format(now(), "yyyy-mm-dd")
            save_dir = "plots/capital_distribution/$(date_str)"
            mkpath(save_dir)
            Plots.savefig("$(save_dir)/episode_$(i).png")
        end

        push!(total_rewards, mean(episode_rewards))
    end 
    
    Plots.savefig("plots/capital_distribution/episodes_full.png")
    plt = Plots.plot(1:NUM_EPISODES, total_rewards, xlabel="Episode", ylabel="total reward", title="DDPG Training")
    Plots.savefig("plots/total_rewards.png")
end