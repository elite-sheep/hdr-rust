// Copyright 2020 Yuchen Wong

use opencv::core::{ CV_8UC3, CV_32FC3, Mat, MatExpr, Point, Vec3f, MatTrait};
use opencv::prelude::{ MatExprTrait };
use opencv::imgproc::{ gaussian_blur, spatial_gradient, COLOR_BGR2GRAY};
use std::error::Error;
use std::vec::Vec;

#[path = "../base/math_utils.rs"] mod math_utils;
#[path = "../base/opencv_utils.rs"] mod opencv_utils;

use math_utils::{PI};
use opencv_utils::{ get_pixel, set_pixel };

pub fn sift_feature_description(src: &Mat,
                                feature_points: &Vec<Point>,
                                feature_mat: &mut Mat) -> Result<(), Box<dyn Error>> {
    let mut gray_image = Mat::default()?;
    opencv::imgproc::cvt_color(src, &mut gray_image, COLOR_BGR2GRAY, 0).unwrap();

    let rows = src.rows();
    let cols = src.cols();

    // Step1: Compute Orientation
    let mut ix: Mat = Mat::default()?;
    let mut iy: Mat = Mat::default()?;
    let mut buffer_x = Mat::default()?;
    let mut buffer_y = Mat::default()?;
    spatial_gradient(&gray_image, &mut buffer_x, &mut buffer_y, 3, BORDER_DEFAULT).unwrap();
    buffer_x.convert_to(&mut ix, CV_32FC1, 1.0, 0.0).unwrap();
    buffer_y.convert_to(&mut iy, CV_32FC1, 1.0, 0.0).unwrap();

    let mut ix2 = Mat::default()?;
    opencv::core::multiply(&ix, &ix, &mut ix2, 1.0, -1).unwrap();
    let mut iy2 = Mat::default()?;
    opencv::core::multiply(&iy, &iy, &mut iy2, 1.0, -1).unwrap();

    let mut buffer = Mat::default()?;
    opencv::core::add_weighted(&ix2, 1.0, &iy2, 1.0, 0.0, &mut buffer,
                               -1).unwrap();
    let mut orientation = Mat::default()?;
    opencv::core::sqrt(&buffer, &mut orientation).unwrap();

    // Step2: Calculate main orientation
    let mut orientation_bins: [Mat, 36];
    let bin_size: f32 = 10.0;
    for i in 0..36 {
        orientation_bins[i] = Mat::zeros(rows, cols, CV_32FC1).unwrap().to_mat().unwrap();
    }

    for i in 0..rows {
        for j in 0..cols {
            let ixx = get_pixel::<f32>(&ix, i, j);
            let iyy = get_pixel::<f32>(&iy, i, j);
            let theta = (iyy / (ixx + 1e-4)).atan() * (180.0 / PI);
            // Convert theta to [0, 360.0]
            if theta < 0.0 {
                theta += 360.0;
            }
            let raw_bin = ((theta + 0.5 * bin_size) / bin_size) % 36;
            let bin = raw_bin as i32;
            set_pixel(&mut orientation_bins[bin], i, j, get_pixel(orientation, i, j));
        }
    }

    let mut buffer = Mat::default()?;
    for i in 0..36 {
        orientation_bins[i].copy_to(&mut buffer).unwrap();
        gaussian_blur(&buffer, &mut orientation_bins[i], 
                      Size::new(9, 9), 3, 0, BORDER_DEFAULT).unwrap();
    }

    let mut main_orientation = Mat::zeros(rows, cols, CV_8UC1).unwrap().to_mat().unwrap();
    for i in 0..rows {
        for j in 0..cols {
            let mut cur_orientation: u8 = 0;
            let mut cur_orientation_value: f32 = 0.0;

            for k in 0..36 {
                let val = get_pixel::<f32>(&orientation_bins[k], i, j);
                if (val > cur_orientation_value) {
                    cur_orientation_value = val;
                    cur_orientation = k;
                }
            }

            set_pixel::<u8>(&mut main_orientation, i, j, cur_orientation);
        }
    }
}
