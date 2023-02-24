module HaloPencils

using MPI
using PencilArrays

include("HaloUpdates.jl")
import .HaloUpdates: update_halo!

export update_halo!
export HaloPencil
export size_local


const ArrayRegion{N} = NTuple{N,UnitRange{Int}} where N
include("data_ranges.jl")

struct HaloPencil{
        N,  # spatial dimensions
        M   # MPI topology dimensions (< N)
    }
    # Underlying pencil configuration from PencilArrays
    pencil :: Pencil{N,M}
    # Global array dimensions (excluding overlapping halos)
    size_global :: Dims{N}
    # Halo size in each decomposed dimension
    halo_levels :: Dims{M}
    # Part of the array held by every process
    axes_all :: Array{ArrayRegion{N}, M}
    # Part of the array held by the local process
    axes_local :: ArrayRegion{N}
    
    function HaloPencil(
        topology::MPITopology,
        size_global::Dims{N},
        halo_levels::Dims{M},
        decomp_dims::Dims{M} = default_decomposition(N, Val(M))
    ) where {M, N}
        procs = complete_dims(Val(N), decomp_dims, topology.dims)
        halos = complete_halos(Val(N), decomp_dims, halo_levels)
        size_extended = size_global .+ 2 .* halos .* procs
        pencil = Pencil(topology, size_extended, decomp_dims)
        axes_all = generate_axes_matrix(decomp_dims, topology.dims, size_global, halo_levels)
        axes_local = axes_all[coords_local(topology)...]
        new{N, M}(
            pencil, size_global, halo_levels, axes_all, axes_local
        )
    end

end

function HaloPencil(dims::Dims, halo_levels::Dims{M}, decomp::Dims{M}, comm::MPI.Comm) where {M}
    topo = MPITopology(comm, Val(M))
    HaloPencil(topo, dims, halo_levels, decomp)
end

HaloPencil(dims::Dims{N}, halo_levels::Dims{M}, comm::MPI.Comm) where {N, M} = 
    HaloPencil(dims, halo_levels, default_decomposition(N, Val(M)), comm)

function default_decomposition(N, ::Val{M}) where {M}
    @assert 0 < M ≤ N
    ntuple(d -> N - M + d, Val(M))
end

size_local(halo_pen::HaloPencil) = size(hpen.pencil)


end