# Functions determining local data ranges in the
# different pencil configurations including a halo

function local_data_range(p, P, N, halo_level::Integer)
    @assert 1 <= p <= P
    a = (N * (p - 1)) ÷ P + 1 - halo_level
    b = (N * p) ÷ P + halo_level
    a:b
end

# "Complete" dimensions not specified in `dims` with ones.
# Examples:
# - if N = 5, dims = (2, 3) and vals = (42, 12), this returns (1, 42, 12, 1, 1).
# - if N = 5, dims = (3, 2) and vals = (42, 12), this returns (1, 12, 42, 1, 1).
function complete_dims(::Val{N}, dims::Dims{M}, vals::Dims{M}) where {N, M}
    @assert N >= M
    vals_all = ntuple(Val(N)) do n
        i = findfirst(==(n), dims)
        if i === nothing
            1  # this dimension is not included in `dims`, so we put a 1
        else
            vals[i]
        end
    end
    vals_all :: Dims{N}
end

# "Complete" halo levels not specified in `dims` with zeros.
# Examples:
# - if N = 5, dims = (2, 3) and vals = (3, 1), this returns (0, 3, 1, 0, 0).
# - if N = 5, dims = (3, 2) and vals = (3, 1), this returns (0, 1, 3, 0, 0).
function complete_halos(::Val{N}, dims::Dims{M}, vals::Dims{M}) where {N, M}
    @assert N >= M
    vals_all = ntuple(Val(N)) do n
        i = findfirst(==(n), dims)
        if i === nothing
            0  # this dimension is not included in `dims`, so we put a 1
        else
            vals[i]
        end
    end
    vals_all :: Dims{N}
end

# Get axes (array regions) owned by all processes in a given pencil
# configuration.
function generate_axes_matrix(
        decomp_dims::Dims{M}, proc_dims::Dims{M}, size_global::Dims{N},
        halo_levels::Dims{M}
    ) where {N, M}
    axes = Array{ArrayRegion{N}, M}(undef, proc_dims)

    # Number of processes in every direction, including those where
    # decomposition is not applied.
    procs = complete_dims(Val(N), decomp_dims, proc_dims)
    halos = complete_halos(Val(N), decomp_dims, halo_levels)

    for I in CartesianIndices(proc_dims)
        coords = complete_dims(Val(N), decomp_dims, Tuple(I))
        axes[I] = local_data_range.(coords, procs, size_global, halos)
    end

    axes
end