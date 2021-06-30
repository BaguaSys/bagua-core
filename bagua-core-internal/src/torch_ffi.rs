// TODO: separate crate for this
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use derivative::Derivative;

pub const DeviceType_CPU: DeviceType = 0;
pub const DeviceType_CUDA: DeviceType = 1;
pub const DeviceType_MKLDNN: DeviceType = 2;
pub const DeviceType_OPENGL: DeviceType = 3;
pub const DeviceType_OPENCL: DeviceType = 4;
pub const DeviceType_IDEEP: DeviceType = 5;
pub const DeviceType_HIP: DeviceType = 6;
pub const DeviceType_FPGA: DeviceType = 7;
pub const DeviceType_MSNPU: DeviceType = 8;
pub const DeviceType_XLA: DeviceType = 9;
pub const DeviceType_Vulkan: DeviceType = 10;
pub const DeviceType_Metal: DeviceType = 11;
pub const DeviceType_XPU: DeviceType = 12;
pub const DeviceType_MLC: DeviceType = 13;
pub const DeviceType_Meta: DeviceType = 14;
pub const DeviceType_HPU: DeviceType = 15;
pub const DeviceType_COMPILE_TIME_MAX_DEVICE_TYPES: DeviceType = 16;
pub type DeviceType = i8;
pub type DeviceIndex = i8;
pub type PyObject = u8;
pub type size_t = ::std::os::raw::c_ulong;

#[repr(C)]
#[derive(Debug)]
pub struct UniquePtr {
    pub _M_t: u8,
}

#[repr(C)]
#[derive(Debug)]
pub struct VariableVersion {
    pub version_counter_: u64,
}
#[repr(C)]
#[derive(Debug)]
pub struct UniqueVoidPtr {
    pub data_: *mut ::std::os::raw::c_void,
    pub ctx_: UniquePtr,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Device {
    pub type_: DeviceType,
    pub index_: DeviceIndex,
}

#[repr(C)]
#[derive(Debug)]
pub struct DataPtr {
    pub ptr_: UniqueVoidPtr,
    pub device_: Device,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct IntrusivePtrTarget {
    pub _bindgen_opaque_blob: [u64; 3usize],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Allocator {
    pub _bindgen_opaque_blob: u64,
}

#[repr(C)]
#[derive(Debug)]
pub struct StorageImpl {
    pub _base: IntrusivePtrTarget,
    pub data_ptr_: DataPtr,
    pub size_bytes_: size_t,
    pub resizable_: bool,
    pub received_cuda_: bool,
    pub allocator_: *mut Allocator,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union SizesAndStrides__bindgen_ty_1 {
    pub outOfLineStorage_: *mut i64,
    pub inlineStorage_: [i64; 10usize],
    _bindgen_union_align: [u64; 10usize],
}

#[repr(C)]
#[derive(Derivative)]
#[derivative(Debug)]
pub struct SizesAndStrides {
    pub size_: size_t,
    #[derivative(Debug = "ignore")]
    pub __bindgen_anon_1: SizesAndStrides__bindgen_ty_1,
}

#[repr(C)]
#[derive(Debug)]
pub struct Storage {
    pub storage_impl_: *mut StorageImpl,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TypeMeta {
    pub index_: u16,
}

#[repr(C)]
#[derive(Debug)]
pub struct CppOptional {
    pub _address: u8,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct __BindgenBitfieldUnit<Storage, Align>
where
    Storage: AsRef<[u8]> + AsMut<[u8]>,
{
    storage: Storage,
    align: [Align; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DispatchKeySet {
    pub repr_: u64,
}

#[repr(C)]
#[derive(Debug)]
pub struct TensorImpl {
    pub _base: IntrusivePtrTarget,
    pub storage_: Storage,
    pub autograd_meta_: UniquePtr,
    pub named_tensor_meta_: UniquePtr,
    pub version_counter_: VariableVersion,
    pub pyobj_: *mut PyObject,
    pub sizes_and_strides_: SizesAndStrides,
    pub storage_offset_: i64,
    pub numel_: i64,
    pub data_type_: TypeMeta,
    pub device_opt_: CppOptional,
    pub _bitfield_1: __BindgenBitfieldUnit<[u8; 1usize], u8>,
    pub storage_access_should_throw_: bool,
    pub _bitfield_2: __BindgenBitfieldUnit<[u8; 1usize], u8>,
    pub key_set_: DispatchKeySet,
}
