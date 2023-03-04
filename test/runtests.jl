using HaloPencils
using MPI
using Test

@testset "HaloPencils.jl" begin
    # Write your tests here.
    for n in [1,2,6]   # Number of processes
        mpiexec() do exe    # MPI wrapper
            run(`$exe -n $n $(Base.julia_cmd()) halo_rank_test.jl`)
            run(`$exe -n $n $(Base.julia_cmd()) grid_index_test.jl`)
        end
        @test true
    end
end