include("nn.jl")
# === TD3 agent (compatible with your Net / step! / back_custom!) ===

mutable struct TD3
    # Actor
    π_       :: Net
    π_target :: Net

    # Twin critics
    Q1_       :: Net
    Q2_       :: Net
    Q1_target :: Net
    Q2_target :: Net

    # Replay buffer: (state, action, reward, next_state, done)
    replay_buffer :: Vector{Tuple{Vector{Float64}, Float64, Float64, Vector{Float64}, Float64}}
    priorities    :: Vector{Float64}

    # Discount / Polyak
    γ   :: Float64
    τ   :: Float64

    # TD3 hyperparams
    policy_delay      :: Int        # e.g., 2
    target_noise_std  :: Float64    # e.g., 0.2 (in action units)
    target_noise_clip :: Float64    # e.g., 0.5

    # Bookkeeping
    step_count :: Int
end

function TD3(π::Net, Q̂::Net, γ::Float64, τ::Float64;
             policy_delay::Int=2, target_noise_std::Float64=0.2, target_noise_clip::Float64=0.5)
    # Build twins from your critic template
    Q1 = deepcopy(Q̂)
    Q2 = deepcopy(Q̂)
    TD3(π, deepcopy(π),
        Q1, Q2, deepcopy(Q1), deepcopy(Q2),
        Tuple{Vector{Float64}, Float64, Float64, Vector{Float64}, Float64}[],
        Float64[], γ, τ,
        policy_delay, target_noise_std, target_noise_clip,
        0)
end

# Soft update (same as your Quant)
function update_target_network!(target_net::Net, main_net::Net, τ::Float64)
    for i in eachindex(main_net.layers)
        target_layer = target_net.layers[i]
        main_layer   = main_net.layers[i]
        target_layer.w .= (1 - τ) * target_layer.w .+ τ * main_layer.w
        target_layer.b .= (1 - τ) * target_layer.b .+ τ * main_layer.b
    end
end

# Experience API (mirrors your Quant)
function add_experience!(td3::TD3, s::Vector{Float64}, a::Float64, r::Float64, s′::Vector{Float64}, d::Float64)
    push!(td3.replay_buffer, (s, a, r, s′, d))
    push!(td3.priorities, 1.0)
    if length(td3.replay_buffer) > 10_000
        popfirst!(td3.replay_buffer)
        popfirst!(td3.priorities)
    end
end

# ∂Q1/∂a at (s, a) using your Net's step!/gradient convention
function dQ1_da(td3::TD3, s::Vector{Float64}, a::Vector{Float64})
    x = vcat(s, a)
    oldL′ = td3.Q1_.L′
    td3.Q1_.L′ = (ŷ, y) -> ones(eltype(ŷ), size(ŷ))  # return vector of ones matching ŷ
    g_in = step!(td3.Q1_, x, [0.0], 0.0, 0.0, 1.0, false)  # no update; gets ∂Q/∂input
    td3.Q1_.L′ = oldL′
    a_dim = td3.π_.output.out_features  # 1 in your setup
    return g_in[end - a_dim + 1:end]    # tail is ∂Q/∂a (as a Vector)
end

# One TD3 training step (all 3 tricks)
function train_td3_step!(td3::TD3, α_Q::Float64, α_π::Float64, λ::Float64, batch_size::Int)
    if length(td3.replay_buffer) < batch_size
        return
    end

    td3.step_count += 1
    mb = [td3.replay_buffer[rand(1:end)] for _ in 1:batch_size]

    # ---- Critics update every step ----
    for (s, a_scalar, r, s′, d) in mb
        a_vec = [a_scalar]  # keep 1-D vectors for Net

        # Target policy smoothing on target action
        a′ = td3.π_target(s′)[1]
        ε  = clamp(td3.target_noise_std * randn(), -td3.target_noise_clip, td3.target_noise_clip)
        a′ = clamp(a′ + ε, 0.0, 1.0)
        a′v = [a′]

        # Twin target critics; NOTE: elementwise min over 1-D vectors
        Q1′ = td3.Q1_target(vcat(s′, a′v))   # Vector{Float64} length 1
        Q2′ = td3.Q2_target(vcat(s′, a′v))
        Qmin′ = min.(Q1′, Q2′)               # <- elementwise (vector-safe)

        # y is also a Vector{Float64} length 1, like in your DDPG code
        y = r .+ td3.γ .* (1 .- d) .* Qmin′

        # Critic 1 step on (s,a)->y
        step!(td3.Q1_, vcat(s, a_vec), y, α_Q, λ, 1/length(mb))
        # Critic 2 step on (s,a)->y
        step!(td3.Q2_, vcat(s, a_vec), y, α_Q, λ, 1/length(mb))
    end

    # ---- Delayed actor + target updates ----
    if (td3.step_count % td3.policy_delay == 0)
        # Actor gradient via Q1
        for (s, _, _, _, _) in mb
            aπ = td3.π_(s)                 # Vector{Float64}
            g  = dQ1_da(td3, s, aπ)        # Vector{Float64}
            back_custom!(td3.π_, s, -g, α_π, λ, 1/length(mb))  # ascend Q1
        end
        # Polyak soft updates
        update_target_network!(td3.π_target,  td3.π_,  td3.τ)
        update_target_network!(td3.Q1_target, td3.Q1_, td3.τ)
        update_target_network!(td3.Q2_target, td3.Q2_, td3.τ)
    end
end
