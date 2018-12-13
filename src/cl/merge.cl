#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 32

int find_diag(__global float* ar_1,
              __global float* ar_2,
              int cur_block_len,
              int diag) {

    int l = max(0, diag - cur_block_len);
    int r = min(diag, cur_block_len);
    while (l < r) {
        int m = (l + r) / 2;
        if (ar_1[m] <= ar_2[diag - m - 1]) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}


__kernel void tmp_sort(__global float* a, unsigned int n) {
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);

    __local float a_local[WORK_GROUP_SIZE];
    a_local[local_id] = a[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        for (int i = 0; i < WORK_GROUP_SIZE; i++) {
            for (int j = i + 1; j < WORK_GROUP_SIZE; j++) {
                if (a_local[i] > a_local[j])  {
                    float tmp = a_local[i];
                    a_local[i] = a_local[j];
                    a_local[j] = tmp;
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    a[global_id] = a_local[local_id];
}


__kernel void merge_sort(__global float* a,
                               __global float* res,
                               unsigned int n,
                               unsigned int cur_block_len)
{
    unsigned int global_id = get_global_id(0);

    int local_block = global_id % (cur_block_len / 4);
    int global_block = 2 * (global_id / (cur_block_len / 4));
    int diag_l = 2 * local_block * 4;
    int diag_r = 2 * (local_block + 1) * 4;

    __global float* block_1 = a + global_block * cur_block_len;
    __global float* block_2 = a + (global_block + 1) * cur_block_len;
    int l_1 = find_diag(block_1, block_2, cur_block_len, diag_l);
    int r_1 = find_diag(block_1, block_2, cur_block_len, diag_r);

    int l_2 = diag_l - l_1;
    int r_2 = diag_r - r_1;

    __global float* res_ar = res + global_block * cur_block_len;
    int ind_1 = l_1;
    int ind_2 = l_2;
    while (ind_1 < r_1 || ind_2 < r_2) {
        if (ind_2 == r_2 || (ind_1 < r_1 && block_1[ind_1] <= block_2[ind_2])) {
            res_ar[ind_1 + ind_2] = block_1[ind_1];
            ind_1++;
        } else {
            res_ar[ind_1 + ind_2] = block_2[ind_2];
            ind_2++;
        }
    }
}