##FUNCTIONS
relu(x) = max(0, x)

cse_loss(y_pred, y_true) = sum((y_pred - y_true).^2)

softmax(x) = exp(x) / sum(exp(x))

mse_loss(y_pred, y_true) = sum((y_pred - y_true).^2)

relu_prime(x) = x > 0 ? 1 : 0

using LinearAlgebra


struct Node
    act::Float64
    weights::Array{Float64}
    bias::Float64
end


struct Layer
    nodes::Array{Node}

    in_features::Int
    out_features::Int
    activation::Function
end

struct Net
    layers::Array{Layer}

    output = layers[end]
end






function init(layer_configs::Array{Tuple{Int, Int, Function}})
    layers = Layer[]
    for (in_features, out_features, activation) in layer_configs
        nodes = [Node(0.0, randn(in_features), randn()) for _ in 1:out_features]
        push!(layers, Layer(nodes, in_features, out_features, activation))
    end
    return Net(layers)
end



function predict(net::Net, x)
    """step. x is an input. NOTE that input layer is before the 1st layer in the network."""

    for l in eachindex(net.layers)
        for n in eachindex(net.layers[l].nodes)
            net.layers[1].nodes[n].act = net.layers[1].activation((l == 1) ? x : [n.act for x in net.layers[l - 1].nodes] ⋅ net.layers[1].nodes[n].weights + net.layers[1].nodes[n].bias)
        end
    end
    return [node.act for node in net.output.nodes]
end



function sgd!(net::Net, x::Array{Float64}, y_true::Array{Float64}, α::Float64, λ::Float64)
    """
    Perform a single step of SGD on the given input x and target y_true.
    Updates the weights and biases of the network in-place.
    
    - net: The neural network.
    - x: Input vector.
    - y_true: Target output vector.
    - α: Learning rate.
    - λ: Regularization factor (L2 penalty).
    """
    # Step 1: Forward Pass
    y_pred = predict(net, x)
    
    # Step 2: Compute Gradients 
    # Initialize the errors for the output layer
    output_errors = [2 * (y_pred[n] - y_true[n]) for n in eachindex(y_pred)]
    
    # Backpropagate the errors through the layers
    for l in length(net.layers):-1:2
        layer = net.layers[l]
        prev_layer = net.layers[l - 1]
        
        # Compute gradients for weights, biases, and propagate errors
        for n in 1:length(layer.nodes)
            node = layer.nodes[n]
            error = output_errors[n] * relu_prime(node.act)
            
            # Update weights with SGD
            for w in 1:length(node.weights)
                gradient = error * prev_layer.nodes[w].act
                gradient += λ * node.weights[w] # L2 regularization
                node.weights[w] -= α * gradient
            end
            
            # Update bias
            node.bias -= α * error
            
            # Accumulate errors for the previous layer
            for p in 1:length(prev_layer.nodes)
                prev_layer.nodes[p].act += error * node.weights[p]
            end
        end
        
        # Update errors for the next layer
        output_errors = [relu_prime(prev_layer.nodes[n].act) * prev_layer.nodes[n].act for n in 1:length(prev_layer.nodes)]
    end
end

function train!(net::Net, data::Array{Tuple{Array{Float64}, Array{Float64}}}, epochs::Int, α::Float64, λ::Float64)
    """
    Train the network using SGD.
    
    - data: A collection of (input, target) pairs.
    - epochs: Number of epochs to train for.
    - α: Learning rate.
    - λ: Regularization factor (L2 penalty).
    """
    for epoch in 1:epochs
        for (x, y_true) in data
            sgd!(net, x, y_true, α, λ)
        end
        println("Epoch $epoch complete")
    end
end
