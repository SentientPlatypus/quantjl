## ACTIVATION FUNCTIONS. PASS IN WHOLE VECTORS.
relu(x) = max.(0, x)
softmax(x) = exp.(x) ./ sum(exp.(x))


∇relu(x) = x .> 0

cse_loss(y_pred, y_true) = sum((y_pred - y_true).^2)
mse_loss(y_pred, y_true) = sum((y_pred - y_true).^2)



mutable struct Layer
    activations::Vector{Float64}
    weights::Matrix{Float64}
    bias::Vector{Float64}

    in_features::Int
    out_features::Int
    activation::Function
end

function Layer(in_features::Int, out_features::Int, activation::Function)
    activations = zeros(out_features)
    weights = randn(out_features, in_features)
    bias = zeros(out_features)

    Layer(activations, weights, bias, in_features, out_features, activation)
end

mutable struct Net
    layers::Array{Layer}
    loss::Function
    output::Layer
end

function Net(dims::Array{Int, 2}, loss::Function)
    Net([Layer(dim...) for dim in dims], loss)
end

function Net(layers::Array{Layer}, loss::Function)
    Net(layers, loss, layers[end])
end




function forward!(net::Net, x::Array{Float64})
    """feedforwards the net, and returns the output activatiosns"""
    @assert length(x) == net.layers[1].in_features

    ##First Layer
    for l in eachindex(net.layers)
        layer_input = (l == 1) ? x : net.layers[l - 1].activations
        net.layers[l].activations = net.layers[l].activation.(net.layers[l].weights * layer_input + net.layers[l].bias)
    end
    net.output.activations
end

function back!(net::Net, y::Array{Float64}, α::Float64, λ::Float64)
    """backpropagates the error and updates the weights Currently Assuming mse_loss, relu"""
    ŷ = net.output.activations

    ##Last Layer
    ∇ = 2 * (ŷ - y)
    net.output.weights -= α * (∇ * net.layers[end - 1].activations' + λ * net.output.weights)
    net.output.bias -= α * ∇


    ##Rest of the layers
    for l in reverse(2:length(net.layers) - 1)
        ∇ = net.layers[l + 1].weights' * ∇
        ∇ = ∇ .* ∇relu(net.layers[l].activations)


        net.layers[l].weights -= α * (∇ * net.layers[l - 1].activations' + λ * net.layers[l].weights)
        net.layers[l].bias -= α * ∇
    end
    ## man julia is so nice
end

function step!(net::Net, x::Array{Float64}, y::Array{Float64}, α::Float64, λ::Float64)
    forward!(net, x)
    back!(net, y, α, λ)
end


