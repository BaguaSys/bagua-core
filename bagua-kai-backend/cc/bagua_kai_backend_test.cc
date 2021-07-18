#include <vector>
#include <iostream>
#include <mutex>
#include <thread>

#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#include "bagua_kai_backend.h"

#define CUDACHECK(cmd)                                                                            \
    do                                                                                            \
    {                                                                                             \
        cudaError_t e = cmd;                                                                      \
        if (e != cudaSuccess)                                                                     \
        {                                                                                         \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

std::pair<void *, cudaEvent_t>
value_pass_H2D(void *src_ptr, size_t bytes, cudaStream_t cuda_stream)
{
    void *device_ptr = 0;
    CUDACHECK(cudaMalloc(&device_ptr, bytes));
    CUDACHECK(cudaMemcpyAsync(device_ptr, src_ptr, bytes, cudaMemcpyHostToDevice, cuda_stream));
    cudaEvent_t cuda_event = NULL;
    CUDACHECK(cudaEventCreateWithFlags(
        &cuda_event, cudaEventBlockingSync | cudaEventDisableTiming));
    CUDACHECK(cudaEventRecord(cuda_event, cuda_stream));

    return {device_ptr, cuda_event};
}

struct AllreduceCallbackContext
{
    std::function<void()> inner;
};

void allreduce_callback(void *ctx_raw_ptr)
{
    std::shared_ptr<AllreduceCallbackContext> ctx(
        static_cast<AllreduceCallbackContext *>(ctx_raw_ptr));
    ctx->inner();
}

void allreduce(
    uintptr_t rank,
    uintptr_t nranks,
    uintptr_t device_id,
    const std::string &master_addr,
    int32_t master_port,
    bool copy_tensors,
    const std::string &autotune_service_addr,
    int32_t autotune_service_port)
{
    CUDACHECK(cudaSetDevice(device_id));
    cudaStream_t cuda_stream;
    CUDACHECK(cudaStreamCreate(&cuda_stream));

    std::vector<std::shared_ptr<void> > memory_holder;
    std::vector<std::vector<bagua::BaguaTensor> > io_tensors;
    const int tensor_num = 10;
    float input_value = device_id;
    for (int i = 0; i < tensor_num; i++)
    {
        auto input = value_pass_H2D(&input_value, sizeof(input_value), cuda_stream);
        float zero = 0;
        auto output = value_pass_H2D(&zero, sizeof(input_value), cuda_stream);

        memory_holder.push_back(std::shared_ptr<void>(input.first, [](void *ptr)
                                                      { CUDACHECK(cudaFree(ptr)); }));
        memory_holder.push_back(std::shared_ptr<void>(output.first, [](void *ptr)
                                                      { CUDACHECK(cudaFree(ptr)); }));

        const std::string &tensor_name = "tensor-" + std::to_string(i);
        io_tensors.push_back({
            bagua::BaguaTensor(
                tensor_name,
                reinterpret_cast<uintptr_t>(device_id),
                input.first,
                1,
                "f32",
                (uint64_t)(input.second)),
            bagua::BaguaTensor(
                tensor_name,
                reinterpret_cast<uintptr_t>(device_id),
                output.first,
                1,
                "f32",
                (uint64_t)(output.second)),
        });
    }

    bagua::BaguaBackendForKAI backend(rank,
                                      nranks,
                                      device_id,
                                      master_addr,
                                      master_port,
                                      copy_tensors);

    std::cerr << "backend init OK" << std::endl;

    std::vector<bagua::BaguaTensor> register_tensors;
    for (const auto &io : io_tensors)
    {
        register_tensors.push_back(io[0]);
    }
    int ret = backend.register_tensors(
        "model-test", register_tensors, autotune_service_addr, autotune_service_port, copy_tensors);
    EXPECT_EQ(ret, 0);

    for (int i = 0; i < io_tensors.size(); i++)
    {
        auto io = io_tensors[i];
        bagua::Tensor output = io[1];
        backend.allreduce(io[0], io[1], 0, allreduce_callback,
                          new AllreduceCallbackContext{
                              [output, cuda_stream, device_id]()
                              {
                                  CUDACHECK(cudaSetDevice(device_id));

                                  float result;
                                  CUDACHECK(cudaMemcpy(&result, output.ptr(), sizeof(result), cudaMemcpyDeviceToHost));

                                  EXPECT_EQ(result, 3.5);
                              }});
    }
}

TEST(BaguaKaiBackend, EndToEnd)
{
    const std::string &master_addr = "127.0.0.1";
    const int master_port = 8123;
    const std::string &autotune_service_addr = "127.0.0.1";
    const int autotune_service_port = 8124;
    const int nranks = 8;
    std::vector<std::vector<int> > processes_gpu_setting{
        {0}, {1, 2}, {3, 4, 5, 6, 7}};

    std::vector<int> child_pids;
    for (std::vector<int> gpu_setting : processes_gpu_setting)
    {
        pid_t c_pid = fork();
        ASSERT_NE(c_pid, -1);

        if (c_pid > 0)
        {
            // parent process
            child_pids.push_back(c_pid);
        }
        else
        {
            std::cerr << "Hello from " << getpid() << std::endl;

            std::vector<std::thread> workers;
            for (int i = 0; i < gpu_setting.size(); i++)
            {
                int device_id = gpu_setting[i];

                workers.push_back(std::thread(allreduce,
                                              i, nranks, i, master_addr, master_port, true,
                                              autotune_service_addr, autotune_service_port));
            }

            for (auto &worker : workers)
            {
                worker.join();
            }

            exit(EXIT_SUCCESS);
        }
    }

    // wait child
    wait(nullptr);
}
