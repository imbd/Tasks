#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/max_prefix_sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r((unsigned int) n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")"
                  << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);
            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();
            ocl::Kernel max_prefix_sum(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
            max_prefix_sum.compile();

            unsigned int workGroupSize = 128;

            gpu::gpu_mem_32i mem_gpu_1;
            mem_gpu_1.resizeN(n);
            gpu::gpu_mem_32i mem_gpu_2;
            mem_gpu_2.resizeN(2 * (n + workGroupSize - 1) / workGroupSize);

            {
                timer t;
                for (int iter = 0; iter < benchmarkingIters; ++iter) {
                    int max_sum = 0;
                    int iter_num = 0;
                    mem_gpu_1.writeN(as.data(), (unsigned int) n);
                    auto current_size = (unsigned int) n;
                    while (current_size > 1) {

                        if (iter_num % 2 == 0) {
                            max_prefix_sum.exec(gpu::WorkSize(workGroupSize,
                                                              (current_size + workGroupSize - 1) / workGroupSize *
                                                              workGroupSize),
                                                mem_gpu_1, iter_num, current_size, mem_gpu_2);
                        } else {
                            max_prefix_sum.exec(gpu::WorkSize(workGroupSize,
                                                              (current_size + workGroupSize - 1) / workGroupSize *
                                                              workGroupSize),
                                                mem_gpu_2, iter_num, current_size, mem_gpu_1);
                        }
                        iter_num++;
                        current_size = (current_size + workGroupSize - 1) / workGroupSize;
                    }

                    if (iter_num % 2 == 0) {
                        mem_gpu_1.readN(&max_sum, 1);
                    } else {
                        mem_gpu_2.readN(&max_sum, 1);
                    }
                    EXPECT_THE_SAME(reference_max_sum, std::max(max_sum, 0), "GPU result should be consistent!");
                    t.nextLap();
                }
                std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
                std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            }

        }
    }
}
