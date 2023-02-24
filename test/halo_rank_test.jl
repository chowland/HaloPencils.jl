using MPI
using PencilArrays
using HaloPencils
using Test

MPI.Init()
dims = (8, 36, 36)
decomp_dims = MPI.Dims_create(MPI.Comm_size(MPI.COMM_WORLD), (0,0))
periodic = (true, false)
comm_cart = MPI.Cart_create(MPI.COMM_WORLD, decomp_dims; periodic=periodic)
topo = MPITopology{2}(comm_cart)
pen = Pencil(topo, dims)

A = PencilArray{Float64}(undef, pen)
rank = MPI.Comm_rank(comm_cart)
fill!(A, rank)

lvlhalo = (2, 1)
update_halo!(A, lvlhalo)

nranks = MPI.Comm_size(MPI.COMM_WORLD)

lvy, lvz = lvlhalo
# Test interior is unchanged
@test all(A[:, 1+lvy:end-lvy, 1+lvz:end-lvz] .== rank)

coords = MPI.Cart_coords(comm_cart)
rank_b = mod(rank - decomp_dims[2], nranks)
rank_t = mod(rank + decomp_dims[2], nranks)

# # Test lower halo
if coords[1]==1 && !periodic[1]
    rank_b = rank
else
    rank_b = mod(rank - decomp_dims[2], nranks)
end
@test all(A[:,1:lvy, 1+lvz:end-lvz] .== rank_b)

# # Test upper halo
if coords[1]+1==decomp_dims[1] && !periodic[1]
    rank_t = rank
else
    rank_t = mod(rank + decomp_dims[2], nranks)
end
@test all(A[:,end-lvy+1:end, 1+lvz:end-lvz] .== rank_t)

# # TO DO: ADD TESTS FOR HALO CORNERS

MPI.Finalize()