include("nn.jl")
include("test/visualize.jl")
using StatsBase


# PROBLEM STATEMENT
# Maximize the return on investment (ROI) by learning a policy that determines whether to hold (0), or long (1) a stock, 
# given the state of the market over the last 20 minutes of multiple technical indicators.

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


function add_experience!(quant::Quant, s, a, r, s′, d)
    """Add an experience tuple to the replay buffer with maximum priority"""
    push!(quant.replay_buffer, (s, a, r, s′, d))
    push!(quant.priorities, 1.0)  # New experiences get max priority
    
    if length(quant.replay_buffer) > 10_000  
        popfirst!(quant.replay_buffer)
        popfirst!(quant.priorities)
    end
end


function train!(quant::Quant, α_Q::Float64, α_π::Float64, λ::Float64, batch_size::Int)
    """Train the Quant agent using a minibatch from the replay buffer. Uses the textbook DDPG algorithm."""
    if length(quant.replay_buffer) < batch_size 
        return  
    end

    # Sample a minibatch
    # each transition computes y = r + γ(1 - d) * Q_target(s′, π_target(s′))
    minibatch = [quant.replay_buffer[rand(1:end)] for _ in 1:batch_size]
    ∂Q∂a = 69.0
    for (s, a, r, s′, d) in minibatch
        Q_target_value = quant.Q_target(vcat(s′, quant.π_target(s′)))

        y = r .+ quant.γ * (1 - d) * Q_target_value


        step!(quant.Q_, vcat(s, a), y, α_Q, λ, 1/length(minibatch)) # Back with MBSE
        
        ∂Q∂a = step!(quant.Q_, vcat(s, quant.π_(s)), y, α_Q, λ, 1/length(minibatch), false) 

        if any(isnan, ∂Q∂a) || any(isinf, ∂Q∂a) || ∂Q∂a == 0.0
            error("invalid gradient: $(∂Q∂a)")
        end

        quant.π_.L′ = (ŷ, y) -> -∂Q∂a[end - quant.π_.output.out_features + 1:end] # GRADIENT ASCENT. 
        #step!(quant.π_, s, [69.420], α, λ, 1/length(minibatch))
        back_custom!(quant.π_, s, -∂Q∂a[end - quant.π_.output.out_features + 1:end], α_π, λ, 1/length(minibatch)) 
    end

    # Update target networks: θ_target ← τ * θ + (1 - τ) θ_target
    update_target_network!(quant.π_target, quant.π_, quant.τ)
    update_target_network!(quant.Q_target, quant.Q_, quant.τ)
end


function train_critic!(quant::Quant, α_Q::Float64, λ::Float64, minibatch)
    bs = length(minibatch)
    for (s, a, r, s′, d) in minibatch
        a′ = quant.π_target(s′)
        Q′ = quant.Q_target(vcat(s′, a′))
        y  = r .+ quant.γ * (1 .- d) .* Q′              # Bellman target
        step!(quant.Q_, vcat(s, a), y, α_Q, λ, 1/bs)    # critic SGD on (s,a)→y
    end
    update_target_network!(quant.Q_target, quant.Q_, quant.τ)   # soft update critic target
end

function dQ_da(quant::Quant, s::Vector{Float64}, a::Vector{Float64})
    x = vcat(s, a)

    # Temporarily set dL/dŷ = 1 so backprop returns ∂Q/∂input
    oldL′ = quant.Q_.L′
    quant.Q_.L′ = (ŷ, y) -> ones(eltype(ŷ), size(ŷ))

    g_in = step!(quant.Q_, x, [0.0], 0.0, 0.0, 1.0, false)  # no update, just gradients
    quant.Q_.L′ = oldL′

    a_dim = quant.π_.output.out_features
    return g_in[end - a_dim + 1:end]   # tail slice is ∂Q/∂a
end

function train_actor!(quant::Quant, α_π::Float64, λ::Float64, minibatch)
    bs = length(minibatch)
    for (s, _, _, _, _) in minibatch
        a = quant.π_(s)
        g = dQ_da(quant, s, a)                           # ∂Q/∂a at (s, π(s))
        back_custom!(quant.π_, s, -g, α_π, λ, 1/bs)      # ascend Q by descending -Q
    end
    update_target_network!(quant.π_target, quant.π_, quant.τ)   # soft update actor target
end

function train_new!(quant::Quant, α_Q::Float64, α_π::Float64, λ::Float64, batch_size::Int;
                update_critic::Bool=true, update_actor::Bool=true)
    # Guard: not enough samples
    if length(quant.replay_buffer) < batch_size
        return
    end

    # Uniform sample (keep your own sampler if you change to PER later)
    minibatch = [quant.replay_buffer[rand(1:end)] for _ in 1:batch_size]

    if update_critic
        train_critic!(quant, α_Q, λ, minibatch)
    end
    if update_actor
        train_actor!(quant, α_π, λ, minibatch)
    end
end





