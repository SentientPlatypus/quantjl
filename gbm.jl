using Random
using Statistics
using Plots
# Function to calculate percent returns
function returns(raw::Vector{Float64})
    return [(raw[i] - raw[i-1]) / raw[i-1] for i in 2:length(raw)]
end

# Main vscore function
function vscore(raw::Vector{Float64}, OBS::Int=60, EPOCH::Int=1000, EXT::Int=20, seed::Int=3)
    v = Float64[]  # Result vector

    for t in OBS:length(raw)-1
        temp = raw[t+1-OBS : t+1]
        ret = returns(temp)
        s0 = temp[end]
        μ = mean(ret)
        σ = std(ret)
        drift = μ + 0.5 * σ^2

        # Simulate paths
        path = [cumsum(randn(EXT)) for _ in 1:EPOCH]  # EPOCH paths of EXT steps each

        sum_exceed = 0
        for i in 1:EPOCH
            for j in 1:EXT
                path[i][j] *= σ
                path[i][j] += drift * j
                path[i][j] = s0 .* exp(path[i][j])
                sum_exceed += (path[i][j] > s0)
            end
        end



        if t == length(raw)-4
            plot(path, title="GBM Path", xlabel="Time", ylabel="Value", label=false)
            savefig("gbm_path_full.png")
        end

        push!(v, sum_exceed / (EPOCH * EXT))
    end

    v = (v .- mean(v)) ./ std(v)
    return v
end
