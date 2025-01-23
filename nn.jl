## ACTIVATION FUNCTIONS. PASS IN WHOLE VECTORS.
softmax(x) = exp.(x) ./ sum(exp.(x))

relu(x) = max.(0, x)
relu′(x) = x .> 0

leaky_relu(x) = max.(0.01 * x, x)
leaky_relu′(x) = broadcast((x) -> x > 0 ? 1.0 : 0.01, x)

sigmoid(x) = 1 ./ (1 .+ exp.(-x))
sigmoid′(x) = sigmoid(x) .* (1 .- sigmoid(x))


## LOSS FUNCTIONS
mse_loss(ŷ, y) = sum((ŷ - y).^2)
mse_loss′(ŷ, y) = 2 * (ŷ - y)


cse_loss(ŷ, y) = -sum(y .* log.(ŷ))  #ŷ is output of softmax function. y is one hot encoded
cse_loss′(ŷ, y) = ŷ - y #IF USING SOFTMAX. ∂L∂ŷ = ŷ - y



mutable struct Layer
    a::Vector{Float64}
    z::Vector{Float64}
    w::Matrix{Float64}
    b::Vector{Float64}

    in_features::Int
    out_features::Int

    σ::Function
    σ′::Function

    ∂w::Matrix{Float64}
    ∂b::Vector{Float64}
end

function Layer(in_features::Int, out_features::Int, σ::Function, σ′::Function)
    a = zeros(out_features)
    z = zeros(out_features)

    w = randn(out_features, in_features) * sqrt(2 / (in_features + out_features))
    b = zeros(out_features)

    ∂w = zeros(out_features, in_features)
    ∂b = zeros(out_features)

    Layer(a, z, w, b, in_features, out_features, σ, σ′, ∂w, ∂b)
end

mutable struct Net
    layers::Array{Layer}
    output::Layer
    L::Function
    L′::Function
end

function Net(layers::Array{Layer}, L::Function, L′::Function)
    Net(layers, layers[end], L, L′)
end


function forward!(net::Net, x::Array{Float64})
    """feedforwards the net, and returns the output activatiosns"""
    @assert length(x) == net.layers[1].in_features

    for l in eachindex(net.layers)
        layer_input = (l == 1) ? x : net.layers[l - 1].a
        net.layers[l].z = net.layers[l].w * layer_input + net.layers[l].b
        net.layers[l].a = net.layers[l].σ(net.layers[l].z)
    end
    net.output.a
end

function (Net::Net)(x::Array{Float64})
    forward!(Net, x)
end

function back!(net::Net, x::Array{Float64}, y::Array{Float64}, α::Float64, λ::Float64, B ::Float64=1.0)
    ŷ = net.output.a

    ∂L∂ŷ = net.L′(ŷ, y) * B # Scale gradients
    ∂ŷ∂z = net.output.σ′(net.output.z)
    ∂z∂w = net.layers[end - 1].a'

    ∂L∂z = ∂L∂ŷ .* ∂ŷ∂z 

    net.output.∂w = ∂L∂z * ∂z∂w + λ * net.output.w
    net.output.∂b = ∂L∂z

    net.output.w -= α * net.output.∂w
    net.output.b -= α * net.output.∂b

    for l in (length(net.layers) - 1):-1:1
        # ∂L∂a = ∑(∂z∂a_l * ∂L∂a_j * ∂aj∂z_j) REMEMBER: [∂a∂z_j * ∂L∂a_j stored in ∂L∂z]
        ∂L∂a = net.layers[l + 1].w' * ∂L∂z  #sum handled by matrix *
        ∂a∂z = net.layers[l].σ′(net.layers[l].z)  
        ∂L∂z = ∂L∂a .* ∂a∂z 

        layer_input = (l == 1) ? x : net.layers[l - 1].a
        ∂z∂w = layer_input'  # Input to the current layer

        net.layers[l].∂w = ∂L∂z * ∂z∂w + λ * net.layers[l].w
        net.layers[l].∂b = ∂L∂z

        net.layers[l].w -= α * net.layers[l].∂w
        net.layers[l].b -= α * net.layers[l].∂b
    end

    return net.layers[1].w' * ∂L∂z #return gradients wrt. x (input)
end


function step!(net::Net, x::Array{Float64}, y::Array{Float64}, α::Float64, λ::Float64, B::Float64=1.0)
    forward!(net, x)
    return back!(net, x, y, α, λ, B)
end