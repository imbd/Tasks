#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP 128

__kernel void max_prefix_sum(__global int* data,
                             unsigned int iteration_number,
                             unsigned int size,
                             __global int* result)
{
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int local_size = get_local_size(0);

    __local int ar_max[2 * WORK_GROUP];
    __local int ar_sum[2 * WORK_GROUP];

    ar_max[local_id] = 0;
    ar_sum[local_id] = 0;
    if (global_id < size) {
        ar_max[local_id] = data[global_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (iteration_number == 0) {
        ar_sum[local_id] = ar_max[local_id];
    } else {
        if (global_id < size) {
            ar_sum[local_id] = data[global_id + size];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int start_ind = 0;
    for (int cur_size = WORK_GROUP; cur_size > 1; cur_size /= 2) {
        int ind_1 = start_ind + 2 * local_id;
        int ind_2 = start_ind + 2 * local_id + 1;
        int next_ind = start_ind + cur_size + local_id;

        if (ind_2 < start_ind + cur_size) {
            ar_sum[next_ind] = ar_sum[ind_1] + ar_sum[ind_2];
            ar_max[next_ind] = ar_max[ind_1];
            int new_value = ar_max[ind_2] + ar_sum[ind_1];
            if (ar_max[next_ind] < new_value) {
                ar_max[next_ind] = new_value;
            }
        }
        start_ind += cur_size;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    unsigned int group_id = get_group_id(0);
    result[group_id] = ar_max[start_ind];
    result[group_id + (size + local_size - 1) / local_size] = ar_sum[start_ind];
}