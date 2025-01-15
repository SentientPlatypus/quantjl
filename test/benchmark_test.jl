# FILE: test/nn_test.jl
using Test
include("../nn.jl")  # Assuming your neural network code is in a module named NN

# Define a simple CNOT problem
function generate_cnot_data()
    x = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]
    y = [0.0 0.0; 0.0 1.0; 1.0 1.0; 1.0 0.0]
    return x, y
end

# Initialize a simple neural network
function init_simple_net()
    layers = [
        Layer(2, 2, relu),  # Input layer
        Layer(2, 2, relu)   # Output layer
    ]
    return Net(layers, mse_loss)
end

# Test the neural network on the CNOT problem
@testset "Neural Network CNOT Test" begin
    x, y = generate_cnot_data()
    net = init_simple_net()
    α = 0.01
    λ = 0.01

    for epoch in 1:1000
        for i in 1:size(x, 1)
            step!(net, x[i, :], y[i, :], α, λ)
        end
    end

    print("finished epochs!")

    for i in 1:size(x, 1)
        forward!(net, x[i, :])
        @test isapprox(net.layers[end].activations, y[i, :], atol=0.1)
    end
end