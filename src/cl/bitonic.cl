#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void bitonic_global(__global float *as,
                             int size,
                             int block_size,
                             unsigned int n) {
    unsigned int global_id = get_global_id(0);
    bool is_asc = false;
    if (global_id % (2 * block_size) < block_size) {
        is_asc = true;
    }

    if (global_id + size < n && global_id % (2 * size) < size) {
        float el_1 = as[global_id];
        float el_2 = as[global_id + size];
        if (((el_1 > el_2) && is_asc) || ((el_1 <= el_2) && !is_asc)) {
            as[global_id] = el_2;
            as[global_id + size] = el_1;
        }
    }
}


__kernel void bitonic_local(__global float *as,
                            int size,
                            int block_size,
                            unsigned int n) {
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);

    __local float as_local[WORK_GROUP_SIZE];

    if (global_id < n) {
        as_local[local_id] = as[global_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    bool is_asc = false;
    if (global_id % (2 * block_size) < block_size) {
        is_asc = true;
    }

    for (int cur_size = size; cur_size >= 1; cur_size /= 2) {
        if (global_id + cur_size < n && global_id % (2 * cur_size) < cur_size) {
            float el_1 = as_local[local_id];
            float el_2 = as_local[local_id + cur_size];
            if (((el_1 > el_2) && is_asc) || ((el_1 <= el_2) && !is_asc)) {
                as_local[local_id] = el_2;
                as_local[local_id + cur_size] = el_1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_id < n) {
        as[global_id] = as_local[local_id];
    }
}
