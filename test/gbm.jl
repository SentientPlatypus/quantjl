include("../gbm.jl")
include("../data.jl")
using Test
using Random
using Plots



@testset "GBM NEW" begin
    Random.seed!(3)
    percent_change = get_historical_raw("AAPL")
    gbm_path2 = vscore(percent_change)
    plot(gbm_path2, title="GBM Path", xlabel="Time", ylabel="Value")
    savefig("plots/gbm_path2.png")
end


