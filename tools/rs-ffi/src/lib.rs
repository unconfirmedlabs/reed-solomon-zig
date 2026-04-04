use reed_solomon_simd::{ReedSolomonDecoder, ReedSolomonEncoder};
use std::ptr;

pub struct Enc {
    inner: ReedSolomonEncoder,
    shard_bytes: usize,
}

pub struct Dec {
    inner: ReedSolomonDecoder,
    shard_bytes: usize,
}

#[no_mangle]
pub extern "C" fn rs_encoder_new(original_count: usize, recovery_count: usize, shard_bytes: usize) -> *mut Enc {
    match ReedSolomonEncoder::new(original_count, recovery_count, shard_bytes) {
        Ok(inner) => Box::into_raw(Box::new(Enc { inner, shard_bytes })),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn rs_encoder_add_original(enc: *mut Enc, data: *const u8, len: usize) -> i32 {
    let enc = unsafe { &mut *enc };
    let shard = unsafe { std::slice::from_raw_parts(data, len) };
    match enc.inner.add_original_shard(shard) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn rs_encoder_encode(enc: *mut Enc, out: *mut u8) -> i32 {
    let enc = unsafe { &mut *enc };
    let result = match enc.inner.encode() {
        Ok(result) => result,
        Err(_) => return -1,
    };

    let mut offset = 0usize;
    for shard in result.recovery_iter() {
        unsafe {
            std::ptr::copy_nonoverlapping(shard.as_ptr(), out.add(offset), enc.shard_bytes);
        }
        offset += enc.shard_bytes;
    }
    0
}

#[no_mangle]
pub extern "C" fn rs_encoder_free(enc: *mut Enc) {
    if !enc.is_null() {
        unsafe {
            drop(Box::from_raw(enc));
        }
    }
}

#[no_mangle]
pub extern "C" fn rs_decoder_new(original_count: usize, recovery_count: usize, shard_bytes: usize) -> *mut Dec {
    match ReedSolomonDecoder::new(original_count, recovery_count, shard_bytes) {
        Ok(inner) => Box::into_raw(Box::new(Dec { inner, shard_bytes })),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn rs_decoder_add_original(dec: *mut Dec, index: usize, data: *const u8, len: usize) -> i32 {
    let dec = unsafe { &mut *dec };
    let shard = unsafe { std::slice::from_raw_parts(data, len) };
    match dec.inner.add_original_shard(index, shard) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn rs_decoder_add_recovery(dec: *mut Dec, index: usize, data: *const u8, len: usize) -> i32 {
    let dec = unsafe { &mut *dec };
    let shard = unsafe { std::slice::from_raw_parts(data, len) };
    match dec.inner.add_recovery_shard(index, shard) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn rs_decoder_decode(dec: *mut Dec, out: *mut u8, indices: *mut usize, capacity: usize) -> isize {
    let dec = unsafe { &mut *dec };
    let result = match dec.inner.decode() {
        Ok(result) => result,
        Err(_) => return -1,
    };

    let mut count = 0usize;
    for (index, shard) in result.restored_original_iter() {
        if count >= capacity {
            return -1;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(shard.as_ptr(), out.add(count * dec.shard_bytes), dec.shard_bytes);
            *indices.add(count) = index;
        }
        count += 1;
    }

    count as isize
}

#[no_mangle]
pub extern "C" fn rs_decoder_free(dec: *mut Dec) {
    if !dec.is_null() {
        unsafe {
            drop(Box::from_raw(dec));
        }
    }
}
