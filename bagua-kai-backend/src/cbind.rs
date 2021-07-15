use crate::backend::BaguaSingleBackendForKAI;
use bagua_core_internal::{
    datatypes::{BaguaBucket, BaguaTensor, BaguaTensorDtype},
};

pub fn cstr_to_str(c_s: *const c_char, size: usize) -> &'static str {
    unsafe { str::from_utf8_unchecked(slice::from_raw_parts(c_s as *const u8, size)) }
}

pub struct BaguaTensorC {
    inner: BaguaTensor,
}

#[no_mangle]
pub extern "C" fn bagua_tensor_c_create(
    ptr: u64,
    num_elem: usize,
    num_elem_allocated: usize,
    dtype_ptr: *const c_char,
    dtype_size: usize,
    device_id: usize,
) -> *mut BaguaTensorC {
    let obj = BaguaTensorC {
        inner: BaguaTensor::new(
            ptr,
            num_elem,
            num_elem_allocated,
            cstr_to_str(dtype_ptr, dtype_size),
            device_id,
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
    inplace: bool,
    align_bytes: usize,
) -> *mut BaguaBucketC {
    let tensor_ptr_slice: &[*mut BaguaTensorC] =
        unsafe { slice::from_raw_parts(tensors_ptr, tensors_len) };
    let mut tensors: Vec<&BaguaTensor> = Default::default();
    unsafe {
        for tensor_ptr in tensor_ptr_slice.iter() {
            tensors.push(&((*(*tensor_ptr)).inner));
        }
    };

    let new_bucket = BaguaBucket::new(
        tensors.as_slice(),
        cstr_to_str(name_ptr, name_size),
        inplace,
        align_bytes,
    );
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
) -> *mut BaguaSingleBackendForKAIC {
    let obj = BaguaSingleBackendForKAIC {
        inner: Arc::new(Mutex::new(BaguaSingleBackendForKAI::new(
            rank, nranks, device_id, 
            cstr_to_str(master_addr_ptr, master_addr_size),
            master_port,
        ))),
    };

    // into_raw turns the Box into a *mut, which the borrow checker
    // ignores, without calling its destructor.
    Box::into_raw(Box::new(obj));
}

#[no_mangle]
pub extern "C" fn bagua_single_backend_for_kai_c_destory(
    ptr: &mut *mut BaguaSingleBackendForKAIC,
) {
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
) -> i32 {
    if ptr.is_null() {
        return -1;
    }

    let mut tensors = Vec::new();
    unsafe {
        let slice: &[*mut BaguaTensorC] = slice::from_raw_parts(tensors_ptr, tensors_len);
        for tensor_ptr in slice.iter() {
            tensors.push(&((*(*tensor_ptr)).inner));
        }
    }

    unsafe { 
        (*ptr).inner.lock().register_tensors(
            cstr_to_str(model_name_ptr, model_name_size),
            tensors,
            cstr_to_str(autotune_service_addr_ptr, autotune_service_addr_size),
            autotune_service_port,
        );
    };

    return 0;
}

#[no_mangle]
pub extern "C" fn bagua_single_backend_for_kai_c_allreduce(
    ptr: *mut BaguaSingleBackendForKAIC,
    tensor: *mut BaguaTensorC,
    ready_cuda_event_ptr: u64,
    callback: extern fn(),
) -> i32 {
    if ptr.is_null() {
        return -1;
    }

    unsafe {
        (*ptr).inner.lock().allreduce(
            &(*tensor),
            ready_cuda_event_ptr,
            Arc::new(move || {
                callback();
            }),
        );
    }

    return 0;
}
