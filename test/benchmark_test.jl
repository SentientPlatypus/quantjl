# FILE: test/nn_test.jl
using Test
using Random
using Printf
include("../nn.jl")  # Assuming your neural network code is in a module named NN
include("visualize.jl")

# Define a simple CNOT problem
function generate_cnot_data()
    x = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]
    y = [0.0 0.0; 0.0 1.0; 1.0 1.0; 1.0 0.0]
    return x, y
end

# Initialize a simple neural network
function init_simple_net(seed::Int)
    Random.seed!(seed)
    layers = [
        Layer(2, 2, relu, relu′),  # Input layer
        Layer(2, 2, relu, relu′)   # Output layer
    ]
    return Net(layers, mse_loss, mse_loss′)
end

# Test the forward! method
@testset "Neural Network Forward Test" begin
    x, y = generate_cnot_data()
    net = init_simple_net(1)

    for i in 1:size(x, 1)
        forward!(net, x[i, :])
        @test !isnothing(net.layers[end].a)
    end
end

@testset "Neural Network Seed Test" begin
    x, y = generate_cnot_data()
    net1 = init_simple_net(1)
    net2 = init_simple_net(1)

    # Compare the final weights and biases
    for l in eachindex(net1.layers)
        @test isapprox(net1.layers[l].w, net2.layers[l].w, atol=1e-5)
        @test isapprox(net1.layers[l].b, net2.layers[l].b, atol=1e-5)
    end
end


@testset "Visualize Neural Network Test" begin
    x, y = generate_cnot_data()
    net = init_simple_net(1)
    visualize_net(net, x[1, :])
end

# Test the neural network on the CNOT problem
@testset "Neural Network CNOT Test" begin
    x, y = generate_cnot_data()
    net = init_simple_net(1)
    α = 0.0001
    λ = 0.01


    for epoch in 1:10000
        for i in 1:size(x, 1)
            step!(net, x[i, :], y[i, :], α, λ)
        end
        if epoch % 100 == 0
            loss = 0.0
            for j in 1:size(x, 1)
                forward!(net, x[j, :])
                loss += net.loss(net.layers[end].a, y[j, :])
            end
            println("Epoch: $epoch, Loss: $loss")
        end
    end

    print("finished epochs!")

    for i in 1:size(x, 1)
        forward!(net, x[i, :])
        @test isapprox(net.layers[end].a, y[i, :], atol=0.2)
    end
end


