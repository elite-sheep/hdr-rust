// Copyright 2020 Yuchen Wong

use opencv::core::{ CV_8UC3, Mat, Vec3b };
use opencv::imgproc::{ corner_harris };
use opencv::prelude::{ MatTrait, Vector };
use std::error::Error;

#[path = "../base/opencv_utils.rs"] mod opencv_utils;

use opencv_utils::{ cvt_rgb_image_to_grey };

pub fn harris_detect_corner(src: &Mat,
                            dst: &mut Mat,
                            block_size: i32,
                            ksize: i32,
                            k: f64,
                            border_type: i32,
                            threshold: u8) -> Result<(), Box<dyn Error>> {
    let mut gray_image: Mat = Mat::default()?;
    cvt_rgb_image_to_grey(src, &mut gray_image).unwrap();

    let mut harris_response: Mat = Mat::default()?;
    corner_harris(&gray_image, &mut harris_response, block_size,
                  ksize, k, border_type).unwrap();
    let mut harris_response_normal: Mat = Mat::default()?;
    let mut harris_response_normal_scaled = Mat::default()?;
    opencv::core::normalize(&harris_response, &mut harris_response_normal, 0.0, 255.0, 32,
                            opencv::core::CV_32FC1, &Mat::default()?).unwrap();
    opencv::core::convert_scale_abs(&harris_response_normal, &mut harris_response_normal_scaled, 1.0, 0.0).unwrap();

    let rows = src.rows();
    let cols = src.cols();

    unsafe {
        dst.create_rows_cols(rows, cols, CV_8UC3)?;
    }

    for i in 0..rows {
        for j in 0..cols {
            let pixel = *harris_response_normal_scaled.at_2d::<u8>(i, j).unwrap();
            if pixel > threshold {
                log::trace!("{} {}", i, j);
                *dst.at_2d_mut::<Vec3b>(i, j).unwrap() = Vec3b::from([255, 0, 0]);
            } else {
                *dst.at_2d_mut::<Vec3b>(i, j).unwrap() = *src.at_2d::<Vec3b>(i, j).unwrap();
            }
        }
    }

    Ok(())
} 
