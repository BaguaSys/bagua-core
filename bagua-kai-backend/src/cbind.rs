use core::ffi::c_void;
use libc::c_char;
use parking_lot::Mutex;
use std::{slice, str, sync::Arc};

use crate::backend::BaguaSingleBackendForKAI;
use bagua_core_internal::datatypes::{BaguaBucket, BaguaTensor, BaguaTensorDtype};

pub fn cstr_to_str(c_s: *const c_char, size: usize) -> &'static str {
    unsafe { str::from_utf8_unchecked(slice::from_raw_parts(c_s as *const u8, size)) }
}

pub fn str_to_bagua_tensor_dtype(dtype: &str) -> Result<BaguaTensorDtype, String> {
    match dtype.to_lowercase().as_str() {
        "f32" => Ok(BaguaTensorDtype::F32),
        "f16" => Ok(BaguaTensorDtype::F16),
        "i64" => Ok(BaguaTensorDtype::I64),
        "u8" => Ok(BaguaTensorDtype::U8),
        _ => Err(format!("Invalid dtype={}", dtype)),
    }
}

pub struct BaguaTensorC {
    inner: BaguaTensor,
}

#[no_mangle]
pub extern "C" fn bagua_tensor_c_create(
    name_ptr: *const c_char,
    name_size: usize,
    device_id: usize,
    ptr: u64,
    num_elem: usize,
    dtype_ptr: *const c_char,
    dtype_size: usize,
    ready_cuda_event_ptr: u64,
) -> *mut BaguaTensorC {
    let dtype_str = cstr_to_str(dtype_ptr, dtype_size).to_string();
    let obj = BaguaTensorC {
        inner: BaguaTensor::new(
            cstr_to_str(name_ptr, name_size).to_string(),
            device_id,
            ptr,
            num_elem,
            str_to_bagua_tensor_dtype(dtype_str.as_str()).unwrap(),
            ready_cuda_event_ptr,
        ),
    };

    Box::into_raw(Box::new(obj))
}

#[no_mangle]
pub extern "C" fn bagua_tensor_c_destroy(ptr: &mut *mut BaguaTensorC) {
    if ptr.is_null() {
        return;
    }

    let _ = unsafe { Box::from_raw(*ptr) };

    *ptr = ::std::ptr::null_mut();
}

pub struct BaguaBucketC {
    inner: BaguaBucket,
}

#[no_mangle]
pub extern "C" fn bagua_bucket_c_create(
    tensors_ptr: *const *mut BaguaTensorC,
    tensors_len: usize,
    name_ptr: *const c_char,
    name_size: usize,
) -> *mut BaguaBucketC {
    let tensor_ptr_slice: &[*mut BaguaTensorC] =
        unsafe { slice::from_raw_parts(tensors_ptr, tensors_len) };
    let mut tensors: Vec<&BaguaTensor> = Default::default();
    unsafe {
        for tensor_ptr in tensor_ptr_slice.iter() {
            tensors.push(&((*(*tensor_ptr)).inner));
        }
    };

    let new_bucket = BaguaBucket::new(tensors.as_slice(), cstr_to_str(name_ptr, name_size));
    let new_bucket = match new_bucket {
        Ok(bucket) => bucket,
        Err(error) => {
            println!("BaguaBucket::new failed, error={:?}", error);
            return std::ptr::null_mut();
        }
    };

    let obj = BaguaBucketC { inner: new_bucket };

    Box::into_raw(Box::new(obj))
}

#[no_mangle]
pub extern "C" fn bagua_bucket_c_destroy(ptr: &mut *mut BaguaBucketC) {
    if ptr.is_null() {
        return;
    }

    let _ = unsafe { Box::from_raw(*ptr) };

    *ptr = ::std::ptr::null_mut();
}

pub struct BaguaSingleBackendForKAIC {
    inner: Arc<Mutex<BaguaSingleBackendForKAI>>,
}

#[no_mangle]
pub extern "C" fn bagua_single_backend_for_kai_c_create(
    rank: usize,
    nranks: usize,
    device_id: usize,
    master_addr_ptr: *const c_char,
    master_addr_size: usize,
    master_port: i32,
    cuda_stream_ptr: u64,
) -> *mut BaguaSingleBackendForKAIC {
    let obj = BaguaSingleBackendForKAIC {
        inner: Arc::new(Mutex::new(BaguaSingleBackendForKAI::new(
            rank,
            nranks,
            device_id,
            cstr_to_str(master_addr_ptr, master_addr_size).to_string(),
            master_port,
            cuda_stream_ptr,
        ))),
    };

    // into_raw turns the Box into a *mut, which the borrow checker
    // ignores, without calling its destructor.
    Box::into_raw(Box::new(obj))
}

#[no_mangle]
pub extern "C" fn bagua_single_backend_for_kai_c_destory(ptr: &mut *mut BaguaSingleBackendForKAIC) {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return;
    }

    // Now we know the pointer is non-null, we can continue. from_raw is the
    // inverse of into_raw: it turns the *mut Dramatic back into a
    // Box<Dramatic>. You must only call from_raw once per pointer.
    let obj: Box<BaguaSingleBackendForKAIC> = unsafe { Box::from_raw(*ptr) };

    // We don't *have* to do anything else; once obj goes out of scope, it will
    // be dropped.  I'm going to drop it explicitly, however, for clarity.
    drop(obj);

    // I am, however, going to null out the `ptr` we were passed just so the
    // calling code is less likely to accidentally re-use the pointer.
    *ptr = ::std::ptr::null_mut();
}

#[no_mangle]
pub extern "C" fn bagua_single_backend_for_kai_c_register_tensors(
    ptr: *mut BaguaSingleBackendForKAIC,
    model_name_ptr: *const c_char,
    model_name_size: usize,
    tensors_ptr: *const *mut BaguaTensorC,
    tensors_len: usize,
    autotune_service_addr_ptr: *const c_char,
    autotune_service_addr_size: usize,
    autotune_service_port: i32,
    copy_tensors: bool,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }

    let mut tensors = Vec::new();
    unsafe {
        let slice: &[*mut BaguaTensorC] = slice::from_raw_parts(tensors_ptr, tensors_len);
        for tensor_ptr in slice.iter() {
            tensors.push(((*(*tensor_ptr)).inner).clone());
        }
    }

    unsafe {
        (*ptr).inner.lock().register_tensors(
            cstr_to_str(model_name_ptr, model_name_size).to_string(),
            tensors,
            cstr_to_str(autotune_service_addr_ptr, autotune_service_addr_size).to_string(),
            autotune_service_port,
            copy_tensors,
        );
    };

    return 0;
}

struct SafeCVoidPtr {
    pub inner: *mut c_void,
}

unsafe impl Send for SafeCVoidPtr {}
unsafe impl Sync for SafeCVoidPtr {}

#[no_mangle]
pub extern "C" fn bagua_single_backend_for_kai_c_allreduce(
    ptr: *mut BaguaSingleBackendForKAIC,
    input_tensor: *mut BaguaTensorC,
    output_tensor: *mut BaguaTensorC,
    ready_cuda_event_ptr: u64,
    callback: extern "C" fn(*mut c_void),
    callback_args: *mut c_void,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }

    let callback_args = SafeCVoidPtr {
        inner: callback_args,
    };

    unsafe {
        (*ptr).inner.lock().allreduce(
            &((*input_tensor).inner),
            &((*output_tensor).inner),
            ready_cuda_event_ptr,
            Arc::new(move || {
                callback(callback_args.inner);
            }),
        );
    }

    return 0;
}
