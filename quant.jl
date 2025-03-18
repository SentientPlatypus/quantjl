include("nn.jl")
include("test/visualize.jl")
using StatsBase


# PROBLEM STATEMENT
# Maximize the return on investment (ROI) by learning a policy that determines whether to short (-1), hold (0), or long (1) a stock, 
# given the state of the market over the last 100 days and the amount of capital available.


# STATE is last 100 days, along with current capital

using Random

mutable struct Quant
    π_ ::Net               # Actor network (policy network)  a ∈ [-1, 0, 1]
    Q_ ::Net               # Critic network (Q-value network) 
    π_target::Net          # Target policy network 
    Q_target::Net          # Target Q-value network 

    replay_buffer::Vector{Tuple{Array{Float64, 1}, Float64, Float64, Array{Float64, 1}, Float64}}  # (state, action, reward, next_state)
    priorities::Vector{Float64}  # Priorities for prioritized experience replay
    γ::Float64             # Discount factor 
    τ::Float64             # Target network update rate 
end

function Quant(π_::Net, Q̂::Net, γ::Float64, τ::Float64)
    π_target = deepcopy(π_)
    Q_target = deepcopy(Q̂)

    Quant(π_, Q̂, π_target, Q_target, [], Float64[], γ, τ)
end


function update_target_network!(target_net::Net, main_net::Net, τ::Float64)
    """Soft update of target network weights: θ_target ← τ * θ + (1 - τ)θ_target"""
    for i in eachindex(main_net.layers)
        target_layer = target_net.layers[i]
        main_layer = main_net.layers[i]

        target_layer.w .= (1 - τ) * target_layer.w .+ τ * main_layer.w
        target_layer.b .= (1 - τ) * target_layer.b .+ τ * main_layer.b
    end
end

function train!(quant::Quant, α::Float64, λ::Float64, batch_size::Int)
    """Train the Quant agent using a minibatch from the replay buffer"""
    if length(quant.replay_buffer) < batch_size 
        return  
    end

    # Sample minibatch with priorities if available
    if !isempty(quant.priorities)
        probs = quant.priorities .^ 0.6  # Alpha parameter for prioritization
        probs ./= sum(probs)
        
        # Sample based on priorities (requires StatsBase)
        indices = sample(1:length(quant.replay_buffer), Weights(probs), batch_size)
        minibatch = [quant.replay_buffer[i] for i in indices]
    else
        # Fall back to random sampling if priorities not initialized
        indices = rand(1:length(quant.replay_buffer), batch_size)
        minibatch = [quant.replay_buffer[i] for i in indices]
    end

    for (idx, (s, a, r, s′, d)) in zip(indices, minibatch)
        # Calculate target Q-value using target networks (DDPG approach)
        next_action = quant.π_target(s′)
        Q_target_value = quant.Q_target(vcat(s′, next_action))
        y = r .+ quant.γ * (1 - d) * Q_target_value

        # Update critic (Q-network)
        step!(quant.Q_, vcat(s, a), y, α, λ, 1/batch_size)
        
        # Calculate the action-value gradient for policy improvement
        ∂Q∂a = step!(quant.Q_, vcat(s, quant.π_(s)), y, α, λ, 1/batch_size, false)

        # Handle invalid gradients more gracefully
        if any(isnan, ∂Q∂a) || any(isinf, ∂Q∂a)
            println("Warning: Invalid gradient detected for sample $idx, skipping update")
            continue
        end

        # Extract the gradient with respect to the action
        action_grad = ∂Q∂a[end - quant.π_.output.out_features + 1:end]
        
        # Update actor (policy network) using the action-value gradient
        quant.π_.L′ = (ŷ, y) -> -action_grad  # DDPG uses gradient ascent on policy
        back_custom!(quant.π_, s, -action_grad, α, λ, 1/batch_size)
        
        # Add gradient clipping after backpropagation
        clip_gradients!(quant.π_, 1.0)
        clip_gradients!(quant.Q_, 1.0)
        
        # Update priorities based on TD error
        if !isempty(quant.priorities)
            Q_pred = quant.Q_(vcat(s, a))
            td_error = abs(y[1] - Q_pred[1])
            quant.priorities[idx] = td_error + 0.01  # Small constant to avoid zero priority
        end
    end

    # Update target networks: θ_target ← τ * θ + (1 - τ) θ_target
    update_target_network!(quant.π_target, quant.π_, quant.τ)
    update_target_network!(quant.Q_target, quant.Q_, quant.τ)
end

function train_old!(quant::Quant, α::Float64, λ::Float64, batch_size::Int)
    """Train the Quant agent using a minibatch from the replay buffer"""
    if length(quant.replay_buffer) < batch_size 
        return  
    end

    # Sample a minibatch
    minibatch = [quant.replay_buffer[rand(1:end)] for _ in 1:batch_size]
    ∂Q∂a = 69.0
    for (s, a, r, s′, d) in minibatch
        Q_target_value = quant.Q_target(vcat(s′, quant.π_target(s′)))

        y = r .+ quant.γ * (1 - d) * Q_target_value


        step!(quant.Q_, vcat(s, a), y, α, λ, 1/length(minibatch)) # Back with MBSE
        
        ∂Q∂a = step!(quant.Q_, vcat(s, quant.π_(s)), y, α, λ, 1/length(minibatch), false) 

        if any(isnan, ∂Q∂a) || any(isinf, ∂Q∂a) || ∂Q∂a == 0.0
            error("invalid gradient: $(∂Q∂a)")
        end

        quant.π_.L′ = (ŷ, y) -> -∂Q∂a[end - quant.π_.output.out_features + 1:end] # GRADIENT ASCENT. 
        #step!(quant.π_, s, [69.420], α, λ, 1/length(minibatch))
        back_custom!(quant.π_, s, -∂Q∂a[end - quant.π_.output.out_features + 1:end], α, λ, 1/length(minibatch)) 
    end

    # Update target networks: θ_target ← τ * θ + (1 - τ) θ_target
    update_target_network!(quant.π_target, quant.π_, quant.τ)
    update_target_network!(quant.Q_target, quant.Q_, quant.τ)
end




function add_experience!(quant::Quant, s, a, r, s′, d)
    """Add an experience tuple to the replay buffer with maximum priority"""
    push!(quant.replay_buffer, (s, a, r, s′, d))
    push!(quant.priorities, 1.0)  # New experiences get max priority
    
    if length(quant.replay_buffer) > 10_000  
        popfirst!(quant.replay_buffer)
        popfirst!(quant.priorities)
    end
end





