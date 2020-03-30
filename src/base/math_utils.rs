/* Copyright 2020 Yuchen Wong */

use opencv::core::{Mat};
use opencv::prelude::MatTrait;
use rand::Rng;
use std::error::Error;

// This file contains some helper functions of math
// It is designed and implemented following c coding style.

#[allow(dead_code)]
pub fn gen_random_integer(min_val: i32,
                          max_val: i32) -> i32 {
    let mut rng = rand::thread_rng();
    return rng.gen_range(min_val, max_val);
}

#[allow(dead_code)]
pub fn hat(val: f32,
           min_val: f32,
           max_val: f32) -> f32 {
    let mid_val = (min_val + max_val) / 2.0;
    if val <= mid_val {
        return val - min_val;
    } else {
        return max_val - val;
    }
}

#[allow(dead_code)]
pub fn get_translation_matrix(dst: &mut Mat,
                              tx: i32,
                              ty: i32)
    -> Result<(), Box<dyn Error>> {

    unsafe {
        dst.create_rows_cols(2, 3, opencv::core::CV_32FC1)?;
    }
    *dst.at_2d_mut::<f32>(0, 0).unwrap() = 1.0;
    *dst.at_2d_mut::<f32>(0, 1).unwrap() = 0.0;
    *dst.at_2d_mut::<f32>(0, 2).unwrap() = tx as f32;
    *dst.at_2d_mut::<f32>(1, 0).unwrap() = 0.0;
    *dst.at_2d_mut::<f32>(1, 1).unwrap() = 1.0;
    *dst.at_2d_mut::<f32>(1, 2).unwrap() = ty as f32;

    Ok(())
}
