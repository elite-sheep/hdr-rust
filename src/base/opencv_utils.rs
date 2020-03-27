/* Copyright 2020 Yuchen Wong */

use opencv::core::{CV_8UC1, Mat, Size_, Vec3b};
use opencv::prelude::MatTrait;
use std::error::Error;

// This file contains some helper function of opencv
// It is designed and implemented following c coding style.

pub fn compute_exclusive_image(src: &Mat, 
                         dst: &mut Mat,
                         offset: u8) 
    -> Result<(), Box<dyn Error>> {
    let rows = src.rows();
    let cols = src.cols();

    cvt_rgb_image_to_grey(src, dst)?;

    let median_pixel_value = find_median(dst);
    let high_bound;
    if 255 - median_pixel_value < offset {
        high_bound = 255;
    } else {
        high_bound = median_pixel_value + offset;
    }
    let low_bound;
    if median_pixel_value < offset {
        low_bound = 0;
    } else {
        low_bound = median_pixel_value - offset;
    }
    for i in 0..rows {
        for j in 0..cols {
            let pixel_value: u8 = *dst.at_2d::<u8>(i, j).unwrap();
            if pixel_value <=high_bound && pixel_value >= low_bound {
                *dst.at_2d_mut::<u8>(i, j).unwrap() = 0;
            } else {
                *dst.at_2d_mut::<u8>(i, j).unwrap() = 255;
            }
        }
    }
    Ok(())
}

pub fn compute_mtb_image(src: &Mat, 
                         dst: &mut Mat) 
    -> Result<(), Box<dyn Error>> {
    let rows = src.rows();
    let cols = src.cols();

    cvt_rgb_image_to_grey(src, dst)?;

    let median_pixel_value = find_median(dst);
    for i in 0..rows {
        for j in 0..cols {
            let pixel_value: u8 = *dst.at_2d::<u8>(i, j).unwrap();
            if pixel_value > median_pixel_value {
                *dst.at_2d_mut::<u8>(i, j).unwrap() = 255;
            } else {
                *dst.at_2d_mut::<u8>(i, j).unwrap() = 0;
            }
        }
    }
    Ok(())
}

fn cvt_rgb_image_to_grey(src: &Mat,
                         dst: &mut Mat) -> Result<(), Box<dyn Error>> {
    let rows = src.rows();
    let cols = src.cols();

    unsafe {
        dst.create_rows_cols(rows, cols, CV_8UC1)?;
    }

    for i in 0..rows {
        for j in 0..cols {
            let pixel_value: Vec3b = *src.at_2d::<Vec3b>(i, j).unwrap();
            let pixel_b = pixel_value[0] as u16;
            let pixel_g = pixel_value[1] as u16;
            let pixel_r = pixel_value[2] as u16;
            *dst.at_2d_mut::<u8>(i, j).unwrap() = mix_rgb_to_gray(pixel_b, pixel_g, pixel_r);
        }
    }

    Ok(())
}

pub fn resize_image_with_default(src: &Mat,
                                 dst: &mut Mat,
                                 fx: f64,
                                 fy: f64) -> Result<(), Box<dyn Error>> {
    opencv::imgproc::resize(src, dst, Size_::default(), 
                            fx, fy, opencv::imgproc::INTER_LINEAR)?;
    Ok(())
}

pub fn warp_affine_with_default(src: &Mat,
                                dst: &mut Mat,
                                transform: &Mat) -> Result<(), Box<dyn Error>> {
    opencv::imgproc::warp_affine(src, dst, transform, Size_::default(), 
                opencv::imgproc::INTER_LINEAR, 
                opencv::core::BORDER_CONSTANT,
                opencv::core::Scalar_::default())?;
    Ok(())
}

pub fn find_median(img: &Mat) -> u8 {
    let mut pixel_hist: [i32; 256] = [0; 256];

    for i in 0..img.rows() {
        for j in 0..img.cols() {
            let pixel_value: u8 = *img.at_2d::<u8>(i, j).unwrap();
            pixel_hist[pixel_value as usize] += 1;
        }
    }

    let median_index: i32 = (img.rows() * img.cols()) >> 1;
    let mut cur_pixel_count: i32 = 0;
    let mut res = 255;

    for i in 0..256 {
        cur_pixel_count += pixel_hist[i];
        if cur_pixel_count >= median_index {
            res = i;
            break;
        }
    }

    return res as u8;
}

// We use the RGB to Gray mapping function
// described in Greg's algorithm.
fn mix_rgb_to_gray(b: u16, g: u16, r: u16) -> u8 {
    return ((19 * b + 183 * g + 54 * r) >> 8) as u8; 
}
