#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define WORK_GROUP_SIZE 256

__kernel void sum(__global const int *xs, int n, __global int *res) {

    int localId = get_local_id(0);
    int globalId = get_global_id(0);

    __local int local_xs[WORK_GROUP_SIZE];
    local_xs[localId] = xs[globalId];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * localId < nvalues) {
            local_xs[localId] += local_xs[localId + nvalues / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0) {
        atomic_add(res, local_xs[0]);
    }
}
