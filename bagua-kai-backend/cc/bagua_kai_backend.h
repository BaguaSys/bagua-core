#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>
#include <vector>
#include <memory>

struct BaguaBucketC;

struct BaguaSingleBackendForKAIC;

struct BaguaTensorC;

extern "C"
{

    BaguaTensorC *bagua_tensor_c_create(const char *name_ptr,
                                        uintptr_t name_size,
                                        uintptr_t device_id,
                                        uint64_t ptr,
                                        uintptr_t num_elem,
                                        const char *dtype_ptr,
                                        uintptr_t dtype_size,
                                        uint64_t ready_cuda_event_ptr);

    void bagua_tensor_c_destroy(BaguaTensorC **ptr);

    BaguaBucketC *bagua_bucket_c_create(BaguaTensorC *const *tensors_ptr,
                                        uintptr_t tensors_len,
                                        const char *name_ptr,
                                        uintptr_t name_size);

    void bagua_bucket_c_destroy(BaguaBucketC **ptr);

    BaguaSingleBackendForKAIC *bagua_single_backend_for_kai_c_create(uintptr_t rank,
                                                                     uintptr_t nranks,
                                                                     uintptr_t device_id,
                                                                     const char *master_addr_ptr,
                                                                     uintptr_t master_addr_size,
                                                                     int32_t master_port,
                                                                     uint64_t cuda_stream_ptr);

    void bagua_single_backend_for_kai_c_destory(BaguaSingleBackendForKAIC **ptr);

    int32_t bagua_single_backend_for_kai_c_register_tensors(BaguaSingleBackendForKAIC *ptr,
                                                            const char *model_name_ptr,
                                                            uintptr_t model_name_size,
                                                            BaguaTensorC *const *tensors_ptr,
                                                            uintptr_t tensors_len,
                                                            const char *autotune_service_addr_ptr,
                                                            uintptr_t autotune_service_addr_size,
                                                            int32_t autotune_service_port,
                                                            bool copy_tensors);

    int32_t bagua_single_backend_for_kai_c_allreduce(BaguaSingleBackendForKAIC *ptr,
                                                     BaguaTensorC *input_tensor,
                                                     BaguaTensorC *output_tensor,
                                                     uint64_t ready_cuda_event_ptr,
                                                     void (*callback)(void *),
                                                     void *callback_args);

} // extern "C"

namespace bagua
{

    class BaguaSingleBackendForKAI
    {
    public:
        BaguaSingleBackendForKAI(
            uintptr_t rank,
            uintptr_t nranks,
            uintptr_t device_id,
            const std::string &master_addr,
            int32_t master_port,
            uint64_t cuda_stream_ptr)
        {
            _backend = std::shared_ptr<BaguaSingleBackendForKAIC>(
                bagua_single_backend_for_kai_c_create(
                    rank,
                    nranks,
                    device_id,
                    master_addr.c_str(),
                    static_cast<uintptr_t>(master_addr.length()),
                    master_port,
                    cuda_stream_ptr),
                [](BaguaSingleBackendForKAIC *ptr)
                {
                    bagua_single_backend_for_kai_c_destory(&ptr);
                });
        }

        BaguaSingleBackendForKAIC *ptr()
        {
            _backend.get();
        }

        int register_tensors(
            const std::string &model_name,
            std::vector<BaguaTensorC *> tensors,
            const std::string &autotune_service_addr,
            int32_t autotune_service_port,
            bool copy_tensors)
        {
            return bagua_single_backend_for_kai_c_register_tensors(
                _backend.get(),
                model_name.c_str(),
                static_cast<uintptr_t>(model_name.length()),
                &tensors.at(0),
                static_cast<uintptr_t>(tensors.size()),
                autotune_service_addr.c_str(),
                static_cast<uintptr_t>(autotune_service_addr.length()),
                autotune_service_port,
                copy_tensors);
        }

        int allreduce(
            BaguaTensorC *input_tensor,
            BaguaTensorC *output_tensor,
            uint64_t ready_cuda_event_ptr,
            void (*callback)(void *),
            void *callback_args)
        {
            return bagua_single_backend_for_kai_c_allreduce(
                _backend.get(),
                input_tensor,
                output_tensor,
                ready_cuda_event_ptr,
                callback,
                callback_args);
        }

    private:
        std::shared_ptr<BaguaSingleBackendForKAIC> _backend;
    };

    class BaguaTensor
    {
    public:
        BaguaTensor(
            const std::string &name,
            uintptr_t device_id,
            uint64_t ptr,
            uintptr_t num_elem,
            const std::string &dtype_str,
            uint64_t ready_cuda_event_ptr)
        {
            _tensor = std::shared_ptr<BaguaTensorC>(
                bagua_tensor_c_create(
                    name.c_str(),
                    static_cast<uintptr_t>(name.length()),
                    device_id,
                    ptr,
                    num_elem,
                    dtype_str.c_str(),
                    static_cast<uintptr_t>(dtype_str.length()),
                    ready_cuda_event_ptr),
                [](BaguaTensorC *p)
                {
                    bagua_tensor_c_destroy(&p);
                });
        }

        BaguaTensorC *ptr()
        {
            _tensor.get();
        }

    private:
        std::shared_ptr<BaguaTensorC> _tensor;
    };

    class BaguaBackendForKAI
    {
    public:
        BaguaBackendForKAI(
            uintptr_t rank,
            uintptr_t nranks,
            uintptr_t device_id,
            const std::string &master_addr,
            int32_t master_port,
            bool copy_tensors) : _backend(rank,
                                            nranks,
                                            device_id,
                                            master_addr,
                                            master_port,
                                            copy_tensors)
        {
        }

        int register_tensors(
            const std::string &model_name,
            std::vector<BaguaTensor> tensors,
            const std::string &autotune_service_addr,
            int32_t autotune_service_port,
            bool copy_tensors)
        {
            std::vector<BaguaTensorC *> tensors_ptr;
            for (BaguaTensor t : tensors)
            {
                tensors_ptr.push_back(t.ptr());
            }

            return _backend.register_tensors(model_name, tensors_ptr, autotune_service_addr, autotune_service_port, copy_tensors);
        }

        int allreduce(
            BaguaTensor input_tensor,
            BaguaTensor output_tensor,
            uint64_t ready_cuda_event_ptr,
            void (*callback)(void *),
            void *callback_args)
        {
            return _backend.allreduce(
                input_tensor.ptr(),
                output_tensor.ptr(),
                ready_cuda_event_ptr,
                callback,
                callback_args);
        }

    private:
        BaguaSingleBackendForKAI _backend;
    };

} // namespace bagua
