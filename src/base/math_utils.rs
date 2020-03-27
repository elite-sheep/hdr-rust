/* Copyright 2020 Yuchen Wong */

use opencv::core::{Mat};
use opencv::prelude::MatTrait;
use std::error::Error;

// This file contains some helper functions of math
// It is designed and implemented following c coding style.

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
