# FILE: test/nn_test.jl
using Test
using Random
using Printf
using UnicodePlots

include("../nn.jl")  # Assuming your neural network code is in a module named NN



# Define a simple CNOT problem
function CNOT(seed::Int)
    x = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]
    y = [0.0 0.0; 0.0 1.0; 1.0 1.0; 1.0 0.0]

    Random.seed!(seed)
    layers = [
        Layer(2, 8, leaky_relu, leaky_relu′),  
        Layer(8, 2, leaky_relu, leaky_relu′)   
    ]

    return x, y, Net(layers, mse_loss, mse_loss′)
end

function BigInput(seed::Int)
    Random.seed!(seed)
    # Create input and output data with larger feature space
    x = rand(1000, 784)  # 1000 samples with 784 features each (like flattened MNIST images)
    y = rand(1000, 10)   # Corresponding output with 10 features for regression tasks

    # Define layers with decreasing size (input -> bottleneck)
    layers = [
        Layer(784, 512, leaky_relu, leaky_relu′),  # First hidden layer
        Layer(512, 256, leaky_relu, leaky_relu′),  # Second hidden layer
        Layer(256, 128, leaky_relu, leaky_relu′),  # Third hidden layer
        Layer(128, 64, leaky_relu, leaky_relu′),   # Fourth hidden layer
        Layer(64, 10, leaky_relu, leaky_relu′)     # Output layer
    ]

    # Return the input, output, and network
    return x, y, Net(layers, mse_loss, mse_loss′)
end


# Define a function with 3 outputs
function ThreeOutput(seed::Int)
    x = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]  # Input data
    y = [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0; 0.0 0.0 0.0]  # Output data (3 outputs)

    Random.seed!(seed)
    layers = [
        Layer(2, 4, leaky_relu, leaky_relu′),  
        Layer(4, 3, leaky_relu, leaky_relu′)   
    ]

    return x, y, Net(layers, mse_loss, mse_loss′)
end


function BinOneHot(seed::Int)
    x = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]  # Input data
    y = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]  # Output data (3 outputs)

    Random.seed!(seed)
    layers = [
        Layer(2, 6, leaky_relu, leaky_relu′),  # Input layer to hidden layer with 4 neurons
        Layer(6, 4, leaky_relu, leaky_relu′)   # Hidden layer to output layer with 3 outputs
    ]

    return x, y, Net(layers, mse_loss, mse_loss′)
end

function AND(seed::Int)
    x = [0.0 0.0; 0.0 1.0; 1.0 0.0; 1.0 1.0]  # Input combinations for AND gate
    y = [0.0; 0.0; 0.0; 1.0]  # Output for AND gate

    Random.seed!(seed)
    layers = [
        Layer(2, 2, leaky_relu, leaky_relu′),  # Input layer
        Layer(2, 1, leaky_relu, leaky_relu′)   # Output layer
    ]

    return x, y, Net(layers, mse_loss, mse_loss′)
end



function test_network(
    net::Net, 
    x::Union{Matrix{Float64}, Vector{Float64}}, 
    y::Union{Matrix{Float64}, Vector{Float64}}, 
    α::Float64, 
    λ::Float64, 
    name::String="Network",
    epochs::Int=100000
    )

    loss_history = []
    epoch_history = []
    
    for epoch in 1:epochs
        for i in 1:size(x, 1)
            step!(net, x[i, :], y[i, :], α, λ)
        end
        if epoch % 100 == 0
            loss = 0.0
            for j in 1:size(x, 1)
                forward!(net, x[j, :])
                loss += net.L(net.output.a, y[j, :])
            end
            
            push!(loss_history, loss)
            push!(epoch_history, epoch)
        end
    end

    plt = UnicodePlots.lineplot(
        epoch_history,
        loss_history,
        xlabel="Epoch",
        ylabel="Loss",
        title="$name Loss",
        width=50,
        xticks=true, 
        yticks=true 
    )
    display(plt)


    for i in 1:size(x, 1)
        println(x[i,:], forward!(net, x[i, :]))
        @test isapprox(net.output.a, y[i, :], atol=0.2)
    end
end

# Test the forward! method
@testset "Neural Network Forward Test" begin
    x, y, net = CNOT(1)

    for i in 1:size(x, 1)
        forward!(net, x[i, :])
        @test !isnothing(net.output.a)
    end
end

@testset "Neural Network Seed Test" begin
    x, y, net1 = CNOT(1)
    x2, y2, net2 = CNOT(1)

    # Compare the final weights and biases
    for l in eachindex(net1.layers)
        @test isapprox(net1.layers[l].w, net2.layers[l].w, atol=1e-5)
        @test isapprox(net1.layers[l].b, net2.layers[l].b, atol=1e-5)
    end
end

# Test the neural network on the CNOT problem
@testset "Neural Network AND Test" begin
    x, y, net = AND(1)
    α = 0.001
    λ = 0.0
    test_network(net, x, y, α, λ, "AND")
end


@testset "Neural Network 3 output Test" begin
    x, y, net = ThreeOutput(1)
    α = 0.0001
    λ = 0.0
    test_network(net, x, y, α, λ, "THREE OUTPUT")
end

@testset "Neural Network binary one hot Test" begin
    x, y, net = BinOneHot(1) #layer size is : 2, 6, 4
    α = 0.0001
    λ = 0.0
    test_network(net, x, y, α, λ, "BINARY ONE HOT ")
end


@testset "Neural Network CNOT Test" begin
    x, y, net = CNOT(1)
    α = 0.0001
    λ = 0.0
    test_network(net, x, y, α, λ, "CNOT")
end


