include("nn.jl")
include("test/visualize.jl")

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
    γ::Float64             # Discount factor 
    τ::Float64             # Target network update rate 
end

function Quant(π_::Net, Q̂::Net, γ::Float64, τ::Float64)
    π_target = deepcopy(π_)
    Q_target = deepcopy(Q̂)

    Quant(π_, Q̂, π_target, Q_target, [], γ, τ)
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

    # Sample a minibatch
    minibatch = [quant.replay_buffer[rand(1:end)] for _ in 1:batch_size]
    ∂Q∂a = 69.0
    for (s, a, r, s′, d) in minibatch
        # println("\n Compute target Q-value: y = r + γ Q̂(s', π_target(s'))")
        Q_target_value = quant.Q_target(vcat(s′, quant.π_target(s′)))
        y = r .+ quant.γ * (1 - d) * Q_target_value
        
        step!(quant.Q_, vcat(s, a), y, α, λ) # Back with MBSE
        
        # println("\n Train actor using policy gradient: ∇ J(π) = ∇ Q̂(s, π(s))")
        ∂Q∂a = step!(quant.Q_, vcat(s, quant.π_(s)), y, α, λ)

        quant.π_.L′ = (ŷ, y) -> ∂Q∂a[end - quant.π_.output.out_features + 1:end]
        # println("L' is now $(quant.π_.L′(0.0,0.0))")
        back!(quant.π_, s, [69.420], α, λ) # Use a dummy target since L′ is overridden
    end

    # Update target networks: θ_target ← τ * θ + (1 - τ) θ_target
    update_target_network!(quant.π_target, quant.π_, quant.τ)
    update_target_network!(quant.Q_target, quant.Q_, quant.τ)
end




function add_experience!(quant::Quant, s, a, r, s′, d)
    """Add an experience tuple to the replay buffer"""
    push!(quant.replay_buffer, (s, a, r, s′, d))
    if length(quant.replay_buffer) > 10_000  # Limit buffer size
        popfirst!(quant.replay_buffer)
    end
end





