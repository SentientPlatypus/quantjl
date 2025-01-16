include("../nn.jl")
using Printf
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

        println("  z:")
        println("  [" * join(format_vector(layer.z), ", ") * "]")

        println("  w:")
        for row in eachrow(layer.w)
            println("  [" * join(format_vector(row), ", ") * "]")
        end

        println("  b:")
        println("  [" * join(format_vector(layer.b), ", ") * "]")

        println("  a:")
        println("  [" * join(format_vector(layer.a), ", ") * "]")
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



@testset "Visualize Neural Network Test" begin
    x, y, net = CNOT(1)
    visualize_net(net, x[1, :])
end