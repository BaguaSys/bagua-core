#include <atomic>
#include <condition_variable>
#include <vector>
#include <list>
#include <chrono>
#include <thread>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "bagua_comm_core_cbind.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

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

namespace bagua
{

    class SingleCommunicator
    {
    public:
        SingleCommunicator(
            uintptr_t rank,
            uintptr_t nranks,
            uintptr_t device_id,
            uint64_t stream_ptr,
            const std::string &nccl_unique_id_str)
        {
            _comm = bagua_single_communicator_c_create(rank, nranks, device_id, stream_ptr,
                                                       nccl_unique_id_str.c_str(), static_cast<uintptr_t>(nccl_unique_id_str.length()));
        }

        ~SingleCommunicator()
        {
            bagua_single_communicator_c_destroy(&_comm);
        }

        int nranks()
        {
            uintptr_t nranks = -1;
            bagua_single_communicator_c_nranks(_comm, &nranks);

            return static_cast<int>(nranks);
        }

        int rank()
        {
            uintptr_t rank = -1;
            bagua_single_communicator_c_rank(_comm, &rank);

            return static_cast<int>(rank);
        }

        BaguaSingleCommunicatorC *ptr()
        {
            return _comm;
        }

    private:
        BaguaSingleCommunicatorC *_comm;
    };

    class BaguaCommOpConfig
    {
    public:
        BaguaCommOpConfig(
            BaguaSingleCommunicatorC *communicator_internode,
            BaguaSingleCommunicatorC *communicator_intranode,
            bool hierarchical = false,
            bool average = true,
            bool scattergather = false,
            const char *compression_ptr = nullptr,
            uintptr_t compression_len = 0,
            bool is_decentralized = false,
            const std::string &peer_selection_mode = "",
            uintptr_t communication_interval = 0)
        {
            _config = bagua_comm_op_config_c_create(communicator_internode, communicator_intranode,
                                                    hierarchical,
                                                    average,
                                                    scattergather,
                                                    compression_ptr,
                                                    compression_len,
                                                    is_decentralized,
                                                    peer_selection_mode.c_str(),
                                                    static_cast<uintptr_t>(peer_selection_mode.length()),
                                                    communication_interval);
        }

        ~BaguaCommOpConfig()
        {
            bagua_comm_op_config_c_destroy(&_config);
        }

        BaguaCommOpConfigC *ptr()
        {
            return _config;
        }

    private:
        BaguaCommOpConfigC *_config;
    };

    class BaguaCommBackend
    {
    public:
        BaguaCommBackend(uintptr_t schedule_channel_cap, uintptr_t device_id)
        {
            _backend = bagua_comm_backend_c_create(schedule_channel_cap, device_id);
        }

        ~BaguaCommBackend()
        {
            bagua_comm_backend_c_destroy(&_backend);
        }

        int register_ordered_buckets(const std::vector<BaguaBucketC *> &buckets)
        {
            return bagua_comm_backend_c_register_ordered_buckets(_backend, &buckets[0], buckets.size());
        }

        int mark_communication_ready(BaguaTensorC *bagua_tensor, uint64_t ready_cuda_event_ptr = 0)
        {
            return bagua_comm_backend_c_mark_communication_ready(_backend, bagua_tensor, ready_cuda_event_ptr);
        }

        int wait_pending_comm_ops()
        {
            return bagua_comm_backend_c_wait_pending_comm_ops(_backend);
        }

    private:
        BaguaCommBackendC *_backend;
    };

    class TensorDeclaration
    {
    public:
        TensorDeclaration(const std::string &name,
                          uintptr_t num_elements,
                          const std::string &dtype)
        {
            _tensor_declaration = bagua_tensor_declaration_c_create(
                name.c_str(), name.size(),
                num_elements,
                dtype.c_str(), dtype.size());
        }

        ~TensorDeclaration()
        {
            bagua_tensor_declaration_c_destroy(&_tensor_declaration);
        }

        TensorDeclarationC *ptr()
        {
            return _tensor_declaration;
        }

        static std::string get_name(TensorDeclarationC *td)
        {
            auto deleter = [&](char *ptr)
            {
                cstring_free(ptr);
            };
            std::unique_ptr<char[], decltype(deleter)> s(bagua_tensor_declaration_c_get_name(td), deleter);

            return std::string(s.get());
        }

        std::string get_name()
        {
            return TensorDeclaration::get_name(_tensor_declaration);
        }

    private:
        TensorDeclarationC *_tensor_declaration;
    };

    class BaguaCommCoreTelemetry
    {
    public:
        BaguaCommCoreTelemetry(const std::string &server_addr_ptr)
        {
            _telemetry = bagua_comm_core_telemetry_c_create(server_addr_ptr.c_str(), server_addr_ptr.size());
        }

        ~BaguaCommCoreTelemetry()
        {
            bagua_comm_core_telemetry_c_destroy(&_telemetry);
        }

        int register_models(
            const std::vector<TensorDeclarationC *> &tensors, std::vector<std::vector<TensorDeclarationC *> > &recommended_buckets)
        {
            std::vector<uintptr_t> ordered_tensors(tensors.size());
            std::vector<uintptr_t> buckets_sizes(tensors.size());

            int ret = bagua_comm_core_telemetry_c_register_models(
                _telemetry, &tensors[0], tensors.size(), &ordered_tensors[0], &buckets_sizes[0]);
            if (ret != 0)
            {
                return ret;
            }

            recommended_buckets.clear();
            int ordered_tensors_id = 0;
            for (int i = 0; i < buckets_sizes.size(); i++)
            {
                int bucket_size = buckets_sizes[i];
                if (bucket_size == 0)
                {
                    continue;
                }

                std::vector<TensorDeclarationC *> bucket(bucket_size);
                for (int j = 0; j < bucket_size; j++)
                {
                    int t_id = ordered_tensors[ordered_tensors_id++];
                    bucket[j] = tensors[t_id];
                }
                recommended_buckets.push_back(bucket);
            }
        }

    private:
        BaguaCommCoreTelemetryC *_telemetry;
    };

    class BaguaTensor
    {
    public:
        BaguaTensor(uint64_t ptr,
                    uintptr_t num_elem,
                    uintptr_t num_elem_allocated,
                    const std::string &dtype_str,
                    uintptr_t device_id)
        {
            _tensor = bagua_tensor_c_create(
                ptr,
                num_elem,
                num_elem_allocated,
                dtype_str.c_str(),
                static_cast<uintptr_t>(dtype_str.length()),
                device_id);
        }

        ~BaguaTensor()
        {
            bagua_tensor_c_destroy(&_tensor);
        }

        BaguaTensorC *ptr()
        {
            return _tensor;
        }

    private:
        BaguaTensorC *_tensor;
    };

    class BaguaBucket
    {
    public:
        BaguaBucket(
            const std::vector<BaguaTensorC *> &bucket_tensors,
            const std::string &name,
            bool inplace,
            uintptr_t align_bytes)
        {
            _bucket = bagua_bucket_c_create(&bucket_tensors[0], bucket_tensors.size(),
                                            name.c_str(), name.size(),
                                            inplace, align_bytes);
        }

        ~BaguaBucket()
        {
            bagua_bucket_c_destroy(&_bucket);
        }

        BaguaBucketC *ptr()
        {
            return _bucket;
        }

        void append_centralized_synchronous_op(
            BaguaSingleCommunicatorC *communicator_internode,
            BaguaSingleCommunicatorC *communicator_intranode,
            bool hierarchical = false,
            bool average = true,
            bool scattergather = false)
        {
            bagua_bucket_c_append_centralized_synchronous_op(
                _bucket, communicator_internode, communicator_intranode, hierarchical, average, scattergather);
        }

    private:
        BaguaBucketC *_bucket;
    };

    std::string tensorflow_dtype_to_bagua_dtype(tensorflow::DataType dtype)
    {
        switch (dtype)
        {
        case DT_FLOAT:
            return "f32";
        case DT_HALF:
            return "f16";
        case DT_UINT8:
            return "u8";
        case DT_INT64:
            return "i64";
        default:
            std::ostringstream oss;
            oss << "Invalid tensor type, type=" << dtype;
            throw std::logic_error(oss.str());
        }
    }

    int tensorflow_dtype_byte_size(tensorflow::DataType dtype)
    {
        switch (dtype)
        {
        case DT_FLOAT:
            return 4;
        case DT_HALF:
            return 2;
        case DT_UINT8:
            return 1;
        case DT_INT64:
            return 8;
        default:
            std::ostringstream oss;
            oss << "Invalid tensor type, type=" << dtype;
            throw std::logic_error(oss.str());
        }
    }

    template <class item>
    class channel
    {
    private:
        std::list<item> queue;
        std::mutex m;
        std::condition_variable cv;
        bool closed;

    public:
        channel() : closed(false) {}
        void close()
        {
            std::unique_lock<std::mutex> lock(m);
            closed = true;
            cv.notify_all();
        }
        bool is_closed()
        {
            std::unique_lock<std::mutex> lock(m);
            return closed;
        }
        void push(const item &i)
        {
            std::unique_lock<std::mutex> lock(m);
            if (closed)
                throw std::logic_error("push to closed channel");
            queue.push_back(i);
            cv.notify_one();
        }
        bool get(item &out, bool wait = true)
        {
            std::unique_lock<std::mutex> lock(m);
            if (wait)
                cv.wait(lock, [&]()
                        { return closed || !queue.empty(); });
            if (queue.empty())
                return false;
            out = queue.front();
            queue.pop_front();
            return true;
        }
    };

    struct TensorInfo
    {
        std::string name;
        size_t num_elements;
        tensorflow::DataType dtype;
        int device_id;
        void *output_buffer = nullptr;
        BaguaTensorC *bagua_tensor = nullptr;

        TensorInfo(
            const std::string &name = "",
            size_t num_elements = 0,
            tensorflow::DataType dtype = DT_FLOAT,
            int device_id = 0,
            void *output_buffer = nullptr,
            BaguaTensorC *bagua_tensor = nullptr) : name(name), num_elements(num_elements), dtype(dtype), device_id(device_id), output_buffer(output_buffer), bagua_tensor(bagua_tensor)
        {
        }

        size_t byte_size() {
            return num_elements * bagua::tensorflow_dtype_byte_size(dtype);
        }
    };

} // namespace bagua

struct BaguaGlobalState
{
    static BaguaGlobalState &get_instance()
    {
        static BaguaGlobalState instance;
        return instance;
    }

    ~BaguaGlobalState()
    {
        async_task_queue.close();
    }

    std::atomic<int> init_tensor_count;
    std::mutex register_ordered_buckets_mutex;
    std::unordered_map<std::string, bagua::TensorInfo> tensor_info_dict;
    std::condition_variable register_over_cv;

    std::once_flag bagua_init_flag;
    std::shared_ptr<bagua::SingleCommunicator> comm;
    std::shared_ptr<bagua::BaguaCommBackend> bagua_backend;
    std::shared_ptr<bagua::BaguaCommCoreTelemetry> bagua_comm_core_telemetry;

    std::shared_ptr<tensorflow::PersistentTensor> output_buffer;
    std::vector<std::shared_ptr<tensorflow::PersistentTensor> > output;
    std::vector<std::shared_ptr<bagua::TensorDeclaration> > td_holder;
    std::vector<std::shared_ptr<bagua::BaguaTensor> > bagua_tensors_holder;
    std::vector<std::shared_ptr<bagua::BaguaBucket> > bagua_buckets_holder;

    std::vector<tensorflow::mutex> output_ref_mutex;

    OpKernelContext *op_context{nullptr};
    cudaStream_t bagua_cuda_stream;
    bagua::channel<std::function<void()> > async_task_queue;
    bagua::channel<std::shared_ptr<bagua::TensorDeclaration>> tensor_register_queue;
};

namespace bagua {

    int register_ordered_buckets(OpKernelContext *context, const std::vector<TensorDeclarationC *> &td_list)
    {
        auto &global = BaguaGlobalState::get_instance();

        std::vector<std::vector<TensorDeclarationC *> > td_buckets;
        global.bagua_comm_core_telemetry->register_models(td_list, td_buckets);

        int count = 0;
        void *buffer_ptr = (void *)(global.output_buffer->AccessTensor(context)->tensor_data().data());
        std::vector<BaguaBucketC *> buckets;
        for (int bucket_id = 0; bucket_id < td_buckets.size(); bucket_id++)
        {
            const auto &td_bucket = td_buckets[bucket_id];
            std::vector<BaguaTensorC *> bucket;
            for (auto td_ptr : td_bucket)
            {
                const auto& tensor_name = bagua::TensorDeclaration::get_name(td_ptr);
                auto &ti = global.tensor_info_dict[tensor_name];

                auto bagua_tensor = std::make_shared<bagua::BaguaTensor>(
                        reinterpret_cast<uint64_t>(buffer_ptr),
                        ti.num_elements,
                        ti.num_elements,
                        bagua::tensorflow_dtype_to_bagua_dtype(ti.dtype),
                        ti.device_id);
                global.bagua_tensors_holder.push_back(bagua_tensor);
                bucket.push_back(bagua_tensor->ptr());

                ti.output_buffer = buffer_ptr;
                ti.bagua_tensor = bagua_tensor->ptr();

                buffer_ptr += ti.byte_size();
                count += 1;
            }

            const std::string &bucket_name = "bucket_" + std::to_string(bucket_id);
            auto bagua_bucket = std::make_shared<bagua::BaguaBucket>(bucket, bucket_name, true, 8);
            global.bagua_buckets_holder.push_back(bagua_bucket);
            buckets.push_back(bagua_bucket->ptr());

            bagua_bucket->append_centralized_synchronous_op(global.comm->ptr(), global.comm->ptr());
        }

        return global.bagua_backend->register_ordered_buckets(buckets);
    }

    void backgroud_loop(OpKernelContext *context, int comm_tensors_num)
    {
        auto &global = BaguaGlobalState::get_instance();

        // Register ordered buckets
        std::vector<TensorDeclarationC *> td_list;
        std::shared_ptr<bagua::TensorDeclaration> td;
        while (td_list.size() < comm_tensors_num && global.tensor_register_queue.get(td, true))
        {
            global.td_holder.push_back(td);
            td_list.push_back(td->ptr());
        }
        register_ordered_buckets(context, td_list);

        // Handling communication tasks
        std::function<void()> action;
        while (global.async_task_queue.get(action))
        {
            action();
        }
    }
} // namespace bagua

// Forward declaration of AsGpuStreamValue
namespace stream_executor
{
    namespace gpu
    {
        cudaStream_t AsGpuStreamValue(Stream *stream);
    } // namespace stream_executor
} // namespace gpu

struct Memcopy
{
    void operator()(cudaStream_t stream_ptr, int size, const void *in, void *out)
    {
        // uint64_t count = sizeof(T) * size;
        int64_t count = size;
        // CUDACHECK(cudaMemcpyAsync(out, in, count, cudaMemcpyDeviceToDevice, stream_ptr));
        CUDACHECK(cudaMemcpy(out, in, count, cudaMemcpyDeviceToDevice));
    }
};

class GroupedAllreduceOp : public AsyncOpKernel
{
public:
    explicit GroupedAllreduceOp(OpKernelConstruction *context)
        : AsyncOpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("rank", &_rank));
        OP_REQUIRES_OK(context, context->GetAttr("nranks", &_nranks));
        OP_REQUIRES_OK(context, context->GetAttr("device_id", &_device_id));
        OP_REQUIRES_OK(context, context->GetAttr("nccl_unique_id_str", &_nccl_unique_id_str));
        OP_REQUIRES_OK(context, context->GetAttr("comm_tensors_num", &_comm_tensors_num));
    }

    void ComputeAsync(OpKernelContext *context, DoneCallback done) override
    {
        // // Unrecommended functions: these are functions that have some
        // // current uses but are not recommended for use, and may go away at
        // // some future major version release.
        // //
        // // The following functions all have versions that return Status
        // // to capture error conditions, and are strongly preferred.
        // Tensor* mutable_output(int index);
        // void set_output(int index, const Tensor& tensor);
        // mutex* input_ref_mutex(int index);
        // void set_output_ref(int index, mutex* mu, Tensor* tensor_for_ref);
        // TensorValue release_output(int index);

        const std::string &node_name = name();
        // std::cerr << "node_name=" << node_name
        //           << ", rank=" << _rank
        //           << std::endl;

        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        // for (int i = 0; i < context->num_inputs(); i++)
        // {
        //     const auto &input = context->input(i);
        //     tensorflow::Tensor *output;
        //     OP_REQUIRES_OK_ASYNC(context, context->allocate_output(i, input.shape(), &output), done);
        // }
        // context->SetStatus(Status::OK());
        // done();
        // return;

        // // This output was marked to not be forwarded either during graph
        // // construction or grappler passes.  Force an allocation and copy input to
        // // output.
        // VLOG(1) << "OpKernelContext set_output index " << index << " tensor "
        //         << tensor.DebugString() << " never_forward " << never_forward
        //         << " params_->forward_from_array[index] "
        //         << params_->forward_from_array[index] << " alloc_attr.scope_id "
        //         << output_alloc_attr(index).scope_id;
        // auto new_tensor = MakeUnique<Tensor>();
        // Status s = allocate_tensor(type, tensor.shape(), new_tensor.get(),
        //                         output_alloc_attr(index));
        // TF_CHECK_OK(s);
        // device()->CopyTensorInSameDevice(&tensor, new_tensor.get(),
        //                                 op_device_context(), [](const Status&) {});
        // outputs_[index] = TensorValue(new_tensor.release());

        // register ordered buckets only once
        // 1. allocate temp buffer
        // 2. make output by temp buffer reference
        // 3. register_ordered_buckets

        auto device_context = context->op_device_context();
        if (device_context == nullptr)
        {
            OP_REQUIRES_OK_ASYNC(context, Status(tensorflow::error::FAILED_PRECONDITION, "nullptr device_context!"), done);
            return;
        }
        cudaStream_t current_stream_ptr = stream_executor::gpu::AsGpuStreamValue(device_context->stream());
        auto &global = BaguaGlobalState::get_instance();

        std::call_once(global.bagua_init_flag, [&, this]()
                       {
                           global.init_tensor_count = 0;
                           global.comm.reset(new bagua::SingleCommunicator(
                               _rank, _nranks, _device_id, reinterpret_cast<uint64_t>(current_stream_ptr), _nccl_unique_id_str));
                           global.bagua_backend.reset(new bagua::BaguaCommBackend(10, _device_id));
                           global.bagua_comm_core_telemetry.reset(new bagua::BaguaCommCoreTelemetry(std::getenv("AUTO_TUNE_SERVER_ADDR")));

                           global.bagua_tensors_holder.clear();
                           global.bagua_buckets_holder.clear();
                           global.output.resize(context->num_inputs());
                           std::vector<BaguaTensorC *> bucket;
                           std::vector<TensorDeclarationC *> td_list;

                           std::int64_t buffer_size = 0;
                           for (int i = 0; i < context->num_inputs(); i++)
                           {
                               auto &input = context->input(i);

                               buffer_size += input.NumElements() * bagua::tensorflow_dtype_byte_size(input.dtype());
                           }
                           {
                               ::tensorflow::TensorShape buffer_shape;
                               buffer_shape.AddDim(buffer_size);
                               global.output_buffer.reset(new tensorflow::PersistentTensor());
                               Status status = context->allocate_persistent(tensorflow::DT_UINT8, buffer_shape, global.output_buffer.get(), nullptr);
                               if (!status.ok())
                               {
                                   throw status;
                               }
                               // On GPU allocation is asynchronous, we need to wait for it to
                               // complete.
                               if (device_context != nullptr)
                               {
                                   device_context->stream()->BlockHostUntilDone();
                               }
                           }

                           auto tmp = std::vector<tensorflow::mutex>(global.output.size());
                           global.output_ref_mutex.swap(tmp);

                           std::thread([this, context]()
                                       { bagua::backgroud_loop(context, _comm_tensors_num); })
                               .detach();
                       });

        // Register ordered buckets:
        // 1. push bagua::TensorDeclaration in global.tensor_register_queue
        // 2. collect all comm op bagua::TensorDeclaration
        // 3. Allocate buffers in the order in which TensorDeclaration is enqueued
        // 4. register_ordered_buckets
        if (global.init_tensor_count < _comm_tensors_num)
        {
            std::lock_guard<std::mutex> guard(global.register_ordered_buckets_mutex);
            std::cerr << "node_name=" << node_name
                        << ", rank=" << _rank
                        << ", init_tensor_count=" << global.init_tensor_count
                        << ", _comm_tensors_num=" << _comm_tensors_num
                        << " across"
                        << std::endl;
            if (global.init_tensor_count < _comm_tensors_num)
            {
                for (int i = 0; i < context->num_inputs(); i++)
                {
                    const std::string tensor_name = node_name + "_" + std::to_string(i + 1) + "of" + std::to_string(context->num_outputs());

                    auto &input = context->input(i);
                    auto input_byte_size = input.NumElements() * bagua::tensorflow_dtype_byte_size(input.dtype());

                    auto td = std::make_shared<bagua::TensorDeclaration>(
                        tensor_name,
                        input.NumElements(),
                        bagua::tensorflow_dtype_to_bagua_dtype(input.dtype()));
                    global.tensor_register_queue.push(td);

                    global.tensor_info_dict[tensor_name] = bagua::TensorInfo(
                        tensor_name,
                        input.NumElements(),
                        input.dtype(),
                        _device_id);
                }
            }
            global.init_tensor_count += context->num_inputs();
        }

        int comm_tensors_num = _comm_tensors_num;
        int rank = _rank;
        int nranks = _nranks;
        int device_id = _device_id;
        auto action = [device_id, context, done, node_name, device_context, current_stream_ptr]()
        {
            CUDACHECK(cudaSetDevice(device_id));

            auto &global = BaguaGlobalState::get_instance();
            auto &tensor_info_dict = global.tensor_info_dict;

            global.op_context = context;

            std::vector<cudaEvent_t> ready_events(context->num_inputs(), nullptr);
            void *allreduce_buffer_ptr = (void *)(global.output_buffer->AccessTensor(context)->tensor_data().data());
            for (int i = 0; i < context->num_inputs(); i++)
            {
                const std::string tensor_name = node_name + "_" + std::to_string(i + 1) + "of" + std::to_string(context->num_outputs());
                auto &input = context->input(i);
                auto input_byte_size = input.NumElements() * bagua::tensorflow_dtype_byte_size(input.dtype());

                auto it = tensor_info_dict.find(tensor_name);
                if (it == tensor_info_dict.end())
                {
                    std::cerr << "tensor_info_dict find failed, tensor_name=" << tensor_name
                              << std::endl;
                    OP_REQUIRES_OK_ASYNC(context, Status(tensorflow::error::UNKNOWN, "tensor_info_dict find failed."), done);
                    return;
                }
                auto &tensor_info = it->second;
                void *allreduce_buffer_ptr = tensor_info.output_buffer;

                cudaError_t err;
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                {
                    std::cerr << "cudaDeviceSynchronize failed: Cuda error " << __FILE__ << ":" << __LINE__
                              << " " << cudaGetErrorString(err)
                              << ", i=" << i
                              << std::endl;
                    exit(EXIT_FAILURE);
                }

                err = cudaPeekAtLastError();
                if (err != cudaSuccess)
                {
                    std::cerr << "cudaPeekAtLastError failed: Cuda error " << __FILE__ << ":" << __LINE__
                              << " " << cudaGetErrorString(err)
                              << ", i=" << i
                              << std::endl;
                    exit(EXIT_FAILURE);
                }

                CUDACHECK(cudaMemcpyAsync(
                    (void *)(allreduce_buffer_ptr),
                    reinterpret_cast<const void *>(input.tensor_data().data()),
                    input_byte_size, cudaMemcpyDeviceToDevice, current_stream_ptr));
                cudaEvent_t cuda_event = NULL;
                CUDACHECK(cudaEventCreateWithFlags(
                    &cuda_event, cudaEventBlockingSync | cudaEventDisableTiming));
                CUDACHECK(cudaEventRecord(cuda_event, current_stream_ptr));

                global.bagua_backend->mark_communication_ready(tensor_info.bagua_tensor, reinterpret_cast<uint64_t>(cuda_event));
            }

            global.bagua_backend->wait_pending_comm_ops();
            // CUDACHECK(cudaStreamSynchronize(global.bagua_cuda_stream));

            auto executor = device_context->stream()->parent();
            auto ready_event = std::make_shared<perftools::gputools::Event>(executor);
            ready_event->Init();
            device_context->stream()->ThenRecordEvent(ready_event.get());
            device_context->stream()->ThenWaitFor(ready_event.get());

            // set output
            for (int i = 0; i < context->num_outputs(); i++)
            {
                const std::string tensor_name = node_name + "_" + std::to_string(i + 1) + "of" + std::to_string(context->num_outputs());

                auto it = tensor_info_dict.find(tensor_name);
                if (it == tensor_info_dict.end())
                {
                    std::cerr << "tensor_info_dict find failed, tensor_name=" << tensor_name
                              << std::endl;
                    OP_REQUIRES_OK_ASYNC(context, Status(tensorflow::error::UNKNOWN, "tensor_info_dict find failed."), done);
                }
                auto &tensor_info = it->second;
                void *allreduce_buffer_ptr = tensor_info.output_buffer;
                // auto output_persistent = global.output[i]->AccessTensor(context);

                // context->set_output(i, *output_persistent);

                const auto &input = context->input(i);
                auto input_byte_size = input.NumElements() * bagua::tensorflow_dtype_byte_size(input.dtype());

                tensorflow::Tensor *output;
                context->allocate_output(i, input.shape(), &output);

                CUDACHECK(cudaMemcpyAsync(
                    (void *)(output->tensor_data().data()),
                    reinterpret_cast<const void *>(allreduce_buffer_ptr),
                    input_byte_size,
                    cudaMemcpyDeviceToDevice, current_stream_ptr));
                // auto& cuda_event = ready_events[i];
                // CUDACHECK(cudaEventCreateWithFlags(
                //     &cuda_event, cudaEventBlockingSync | cudaEventDisableTiming));
                // CUDACHECK(cudaEventRecord(cuda_event, current_stream_ptr));

                // tensorflow::Tensor* output;
                // context->allocate_output(i, input.shape(), &output);
                // Memcopy()(
                //     current_stream_ptr,
                //     input_byte_size,
                //     reinterpret_cast<const void*>(allreduce_buffer_ptr),
                //     (void*)(output->tensor_data().data())
                // );

                // context->set_output(i, input);
            }

            // OP_REQUIRES_OK_ASYNC(context, Status::OK(), done);
            context->SetStatus(Status::OK());
            done();
        };
        global.async_task_queue.push(action);
    }

private:
    int _rank;
    int _nranks;
    int _device_id;
    int _comm_tensors_num;
    std::string _nccl_unique_id_str;

    bool _ignore_name_scope;
};

// REGISTER_KERNEL_BUILDER(Name("GroupedAllreduce").Device(DEVICE_CPU),
//                         GroupedAllreduceOp);
REGISTER_KERNEL_BUILDER(Name("GroupedAllreduce").Device(DEVICE_GPU),
                        GroupedAllreduceOp);

REGISTER_OP("GroupedAllreduce")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("rank: int")
    .Attr("nranks: int")
    .Attr("device_id: int")
    .Attr("nccl_unique_id_str: string")
    .Attr("ignore_name_scope: bool = False")
    .Attr("num_tensors: int")
    .Attr("comm_tensors_num: int")
    .Input("tensors: num_tensors*T")
    .Output("sum: num_tensors*T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
    int num_tensors;
    for (int i = 0; i < c->num_inputs(); ++i)
    {
        c->set_output(i, c->input(i));
    }
    return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allreduce on a list tensors. All other processes that do a reduction
on a tensor with the same name must have the same dimension for that tensor.
Tensors are reduced with other tensors that have the same node name for the
allreduce.

Arguments
    tensors:     A list of tensors to reduce.

Output
    sum:    A list of tensors with the same shape as corresponding tensors in `tensors`, summed across all MPI processes.
)doc");
