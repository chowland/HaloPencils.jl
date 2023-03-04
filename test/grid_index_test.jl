using MPI
using HaloPencils
using PencilArrays
using OffsetArrays
using Test

MPI.Init()
dims = (12, 32, 21)
lvlhalo = (1, 2)

proc_dims = MPI.Dims_create(MPI.Comm_size(MPI.COMM_WORLD), (0,0))
periodic = (true, true)
comm_cart = MPI.Cart_create(MPI.COMM_WORLD, proc_dims; periodic=periodic)
topo = MPITopology{2}(comm_cart)
halo_pen = HaloPencil(topo, dims, lvlhalo)

# Global grid vectors
xg = 1:dims[1]
yg = 1:dims[2]
zg = 1:dims[3]
yx = OffsetVector(
    vcat(yg[dims[2]-lvlhalo[1]+1:dims[2]],yg, yg[1:lvlhalo[1]]),
    1-lvlhalo[1]:dims[2]+lvlhalo[1]
)
zx = OffsetVector(
    vcat(zg[dims[3]-lvlhalo[2]+1:dims[3]],zg, zg[1:lvlhalo[2]]),
    1-lvlhalo[2]:dims[3]+lvlhalo[2]
)

X = PencilArray{Float64}(undef, halo_pen.pencil);

function inner(A::AbstractArray, lvl::Dims{2})
    return @view A[:,1+lvl[1]:end-lvl[1],1+lvl[2]:end-lvl[2]]
end

function axes_inner(HP::HaloPencil)
    lvl = HP.halo_levels
    ix = HP.axes_local[1]
    iy = HP.axes_local[2][1+lvl[1]:end-lvl[1]]
    iz = HP.axes_local[3][1+lvl[2]:end-lvl[2]]
    return (ix, iy, iz)
end

inner(X, lvlhalo) .= xg;

update_halo!(X, lvlhalo)

@test all(X .== xg)

ix, iy, iz = axes_inner(halo_pen)
ixx, iyx, izx = halo_pen.axes_local

inner(X, lvlhalo) .= yg[iy]';
update_halo!(X, lvlhalo)

@test all(X .== yx[iyx]')

inner(X, lvlhalo) .= reshape(zg[iz],1,1,:)
update_halo!(X, lvlhalo)
@test all(X .== reshape(zx[izx],1,1,:))