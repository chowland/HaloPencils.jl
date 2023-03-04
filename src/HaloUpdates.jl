module HaloUpdates

export update_halo!

using MPI
using PencilArrays

"""

    update_halo!(A::PencilArray, lvlhalo::Dims{M})

Update the halo-cells of a PencilArray `A`, given that the halo
is of the widths given in `lvlhalo` for each of the decomposed dimensions.

The MPI communicator used for the halo update is that stored in `A`,
which determines whether the domain should be treated as periodic or
not. By default, the MPI communicator created with a PencilArray is
not periodic in either direction. To enable halo updates for a periodic
domain, first create an MPI communicator that specifies the periodicity
and use that to create a PencilArray, as in the following example.

# Example

```julia
dims = (8, 12, 9)
decomp_dims = MPI.Dims_create(MPI.Comm_size(MPI.COMM_WORLD), (0,0))
comm_cart = MPI.Cart_create(MPI.COMM_WORLD, decomp_dims; periodic=(true, true))
topo = MPITopology{2}(comm_cart)
pen = Pencil(topo, dims)

A = PencilArray{Float64}(undef, pen)
rank = MPI.Comm_rank(comm_cart)
fill!(A, rank)

HaloPencils.update_halo!(A, (2,1))
```

---

    update_halo!(A::PencilArray, lvlhalo::Integer)

Convenient `update_halo!` constructor for the case where the halo
width `lvlhalo` is the same in both decomposed dimensions.
"""
function update_halo!(A::PencilArray, lvlhalo::Dims{M}) where M
    comm = get_comm(A)
    dims, periods, coords = MPI.Cart_get(comm)
    periodic = periods.==1

    for (i, n) in enumerate(decomposition(pencil(A)))
        
        tag_b = coords[i]
        if coords[i]+1==dims[i] && periodic[i]
            tag_t = 0
        else
            tag_t = coords[i] + 1
        end

        n_end = size(A)[n]

        # If target process is local, don't use MPI sends
        if tag_t == tag_b
            if periodic[i]
                selectdim(A, n, 1:lvlhalo[i]) .= selectdim(A, n, n_end-2*lvlhalo[i]+1:n_end-lvlhalo[i])
                selectdim(A, n, n_end-lvlhalo[i]+1:n_end) .= selectdim(A, n, 1+lvlhalo[i]:2*lvlhalo[i])
            end
        else

            neigh_b, neigh_t = MPI.Cart_shift(comm, i-1,1)

            T = typeof(A[1,1,1])
            
            # Copy data from PencilArray to send buffers
            send_halo_b = copy(selectdim(A, n, 1+lvlhalo[i]:2*lvlhalo[i]))
            send_halo_t = copy(selectdim(A, n, n_end-2*lvlhalo[i]+1:n_end-lvlhalo[i]))
            recv_halo_b = similar(send_halo_b)
            recv_halo_t = similar(send_halo_t)

            # Receive from bottom
            rreqb = MPI.Irecv!(recv_halo_b, comm; source=neigh_b, tag=tag_b)
            # Receive from top
            rreqt = MPI.Irecv!(recv_halo_t, comm; source=neigh_t, tag=tag_t)
            # Send to bottom
            sreqb = MPI.Isend(send_halo_b, comm; dest=neigh_b, tag=tag_b)
            # Send to top
            sreqt = MPI.Isend(send_halo_t, comm; dest=neigh_t, tag=tag_t)

            MPI.Waitall([sreqb, sreqt, rreqb, rreqt])
            
            if periodic[i] || tag_b > 0
                selectdim(A, n, 1:lvlhalo[i]) .= recv_halo_b
            end
            if tag_t < dims[i]
                selectdim(A, n, n_end-lvlhalo[i]+1:n_end) .= recv_halo_t
            end

        end
    end
end

update_halo!(A::PencilArray, lvl::Integer) = update_halo!(A, (lvl, lvl))

end