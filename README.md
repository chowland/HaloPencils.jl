# HaloPencils

A micro-package for adding halo-cell functionality to [`PencilArrays.jl`](https://github.com/jipolanco/PencilArrays.jl)

Halo updates are performed using the `Irecv` and `Isend` MPI commands, following the Fortran implementation of the [`2decomp-fft`](https://github.com/xcompact3d/2decomp-fft) library.

The only object in the package thus far is a halo update function `update_halo!` that treats the local edges of a `PencilArray` as a halo that needs to be shared with neighbouring processes.

### Future development ideas

- A `HaloPencil` type that wraps a `Pencil`, but also takes in and stores details of the periodicity and the halo size.
    - `localgrid()` functions for a `HaloPencil` that account for the overlap of successive processes due to the presence of a halo.
- A `HaloPencilArray` type wrapping a `PencilArray` that uses a `HaloPencil`
    - `global_view` function that accounts for the halo overlap when indexing (perhaps with an `OffsetArray` even for rank 0 process starting at `1-lvlhalo`)