/* Copyright 2020 Yuchen Wong */

use opencv::core::{CV_8UC1, Mat, Vec3b};
use opencv::imgcodecs::imwrite;
use opencv::prelude::MatTrait;
use opencv::prelude::Vector;
use opencv::types::VectorOfi32;
use std::convert::TryInto;
use std::mem;

// This file contains some helper function of opencv
// It is designed and implemented following c coding style.

pub fn compute_mtb_image(src: &Mat) -> Result<Mat, opencv::Error> {
    log::trace!("Compute MTB image starts.");

    let rows: i32 = src.rows();
    let cols: i32 = src.cols();
    let mut mtb_pixels: Vec<u8> = Vec::new();
    mtb_pixels.reserve((rows*cols).try_into().unwrap());

    for row in 0..rows {
        for col in 0..cols {
            let pixel: &Vec3b = src.at_2d(row, col)?;
            let pixel_b: u32 = pixel[0] as u32;
            let pixel_g: u32 = pixel[1] as u32;
            let pixel_r: u32 = pixel[2] as u32;
            let row_mtb_pixel: u32 = 19 * pixel_b + 183 * pixel_g + 54 * pixel_r;
            mtb_pixels.push((row_mtb_pixel >> 8) as u8);
        }
    }

    let res = Mat::new_rows_cols_with_data(
        rows,
        cols,
        CV_8UC1,
        unsafe { mem::transmute(mtb_pixels.as_ptr())},
        opencv::core::Mat_AUTO_STEP
        );

    let img_write_types = VectorOfi32::with_capacity(0);
    imwrite("/home/yucwang/Desktop/haibara_2_transformed.bmp", &res.unwrap(), &img_write_types);

    return Mat::new_rows_cols_with_data(
        rows,
        cols,
        CV_8UC1,
        unsafe { mem::transmute(mtb_pixels.as_ptr())},
        opencv::core::Mat_AUTO_STEP
        );
}
