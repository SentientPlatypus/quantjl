include("../gbm.jl")
include("../data.jl")
using Test
using Random
using Plots
@testset "GBM" begin
    Random.seed!(3)

    percent_change = get_historical_raw("AAPL")

    # Generate the GBM path
    gbm_path = vscore(percent_change)
    plot(gbm_path, title="GBM Path", xlabel="Time", ylabel="Value")
    savefig("gbm_path.png")
end


