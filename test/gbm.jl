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


@testset "PATH 1000" begin
    Random.seed!(3)
    percent_change = get_historical_raw("AAPL")
    plot(percent_change[begin:1000], title="GBM Path", xlabel="Time", ylabel="Value")
    savefig("plots/appl_1000.png")
end


