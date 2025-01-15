## ACTIVATION FUNCTIONS. PASS IN WHOLE VECTORS.
softmax(x) = exp.(x) ./ sum(exp.(x))

relu(x) = max.(0, x)
relu′(x) = x .> 0

mse_loss(ŷ, y) = sum((ŷ - y).^2)
mse_loss′(ŷ, y) = 2 * (ŷ - y)



mutable struct Layer
    a::Vector{Float64}
    z::Vector{Float64}
    w::Matrix{Float64}
    b::Vector{Float64}

    in_features::Int
    out_features::Int

    σ::Function
    σ′::Function
end

function Layer(in_features::Int, out_features::Int, σ::Function, σ′::Function)
    a = zeros(out_features)
    z = zeros(out_features)
    w = randn(out_features, in_features)
    b = zeros(out_features)

    Layer(a, z, w, b, in_features, out_features, σ, σ′)
end

mutable struct Net
    layers::Array{Layer}
    output::Layer
    L::Function
    L′::Function
end

function Net(dims::Array{Int, 2}, L::Function, L′::Function)
    Net([Layer(dim...) for dim in dims], L, L′)
end

function Net(layers::Array{Layer}, L::Function, L′::Function)
    Net(layers, layers[end], L, L′)
end



function forward!(net::Net, x::Array{Float64})
    """feedforwards the net, and returns the output activatiosns"""
    @assert length(x) == net.layers[1].in_features

    ##First Layer
    for l in eachindex(net.layers)
        layer_input = (l == 1) ? x : net.layers[l - 1].a
        net.layers[l].z = net.layers[l].w * layer_input + net.layers[l].b
        net.layers[l].a = net.layers[l].σ.(net.layers[l].z)
    end
    net.output.a
end

function back!(net::Net, x::Array{Float64}, y::Array{Float64}, α::Float64, λ::Float64)
    """backpropagates the error and updates the w Currently Assuming mse_L, relu"""
    ŷ = net.output.a

    ∂L∂ŷ = net.L′(ŷ, y)
    ∂ŷ∂z = net.output.σ′(net.output.z)
    ∂z∂w = net.layers[end - 1].a'

    partial_∇ = ∂L∂ŷ .* ∂ŷ∂z 

    net.output.w -= α * (partial_∇ * ∂z∂w  +  λ * net.output.w)
    net.output.b -= α * partial_∇

    ## Backpropagate through the hidden layers
    for l in (length(net.layers) - 1):-1:1

        # Gradients for the current layer 
        # ∂L∂a = ∑(∂z∂a_l * ∂L∂a_j * ∂aj∂z_j)  
        # REMEMBER: [∂a∂z_j * ∂z∂a_j stored in partial_∇]
        ∂L∂a = net.layers[l + 1].w' * partial_∇  #sum handled by matrix *
        ∂a∂z = net.layers[l].σ′(net.layers[l].z)  
        partial_∇ = ∂L∂a .* ∂a∂z 

        layer_input = (l == 1) ? x : net.layers[l - 1].a
        ∂z∂w = layer_input'  # Input to the current layer

        net.layers[l].w -= α * (partial_∇ * ∂z∂w + λ * net.layers[l].w)
        net.layers[l].b -= α * partial_∇
    end
end



function step!(net::Net, x::Array{Float64}, y::Array{Float64}, α::Float64, λ::Float64)
    forward!(net, x)
    back!(net, x, y, α, λ)
end

function visualize_net(net::Net, x::Vector{Float64})
    """
    Visualize the neural network with activations, weights, and biases.
    """
    # Forward pass to compute activations
    forward!(net, x)

    # Function to format a vector for display
    function format_vector(v)
        return [@sprintf("% .2f", elem) for elem in v]
    end

    # Function to draw a single layer
    function draw_layer(layer::Layer, index::Int)
        println("Layer $index")

        println("  Activations:")
        println("  [" * join(format_vector(layer.a), ", ") * "]")

        println("  Weights:")
        for row in eachrow(layer.w)
            println("  [" * join(format_vector(row), ", ") * "]")
        end

        println("  Biases:")
        println("  [" * join(format_vector(layer.b), ", ") * "]")

        println("  Outputs:")
        println("  [" * join(format_vector(layer.z), ", ") * "]")
    end

    println("Input:")
    println("[" * join(format_vector(x), ", ") * "]")
    println("\n")

    # Draw each layer
    for (i, layer) in enumerate(net.layers)
        draw_layer(layer, i)
        println("\n")
    end

    # Draw the output layer explicitly (redundant in this architecture but for clarity)
    println("Output Layer:")
    println("  Activations:")
    println("  [" * join(format_vector(net.output.a), ", ") * "]")
end