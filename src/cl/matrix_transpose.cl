#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose(__global float *a, __global float *at, unsigned int m, unsigned int k) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE * TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    tile[local_j * TILE_SIZE + local_i] = a[j * k + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_i < local_j) {
        float tmp = tile[local_j * TILE_SIZE + local_i];
        tile[local_j * TILE_SIZE + local_i] = tile[local_i * TILE_SIZE + local_j];
        tile[local_i * TILE_SIZE + local_j] = tmp;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int ni = get_group_id(0) * TILE_SIZE + local_j;
    int nj = get_group_id(1) * TILE_SIZE + local_i;

    at[ni * k + nj] = tile[local_j * TILE_SIZE + local_i];
}