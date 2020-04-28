// Copyright 2020 Yuchen Wong

use opencv::core::{ CV_8UC1, CV_32FC1, 
    Mat, Point, Point2f, MatTrait, Size, BORDER_DEFAULT};
use opencv::prelude::{ MatExprTrait };
use opencv::imgproc::{ gaussian_blur, spatial_gradient, COLOR_BGR2GRAY};
use std::error::Error;
use std::f32::consts::{PI};
use std::vec::Vec;

#[path = "../base/math_utils.rs"] mod math_utils;
#[path = "../base/opencv_utils.rs"] mod opencv_utils;

use opencv_utils::{ get_pixel, set_pixel };

pub fn sift_feature_description(src: &Mat,
                                feature_points: &Vec<Point>,
                                feature_mat: &mut Mat) -> Result<(), Box<dyn Error>> {
    log::trace!("Starting sift feature descriptor.");

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
    let mut orientation_bins: Vec<Mat> = Vec::new();
    let bin_size: f32 = 10.0;
    for _ in 0..36 {
        orientation_bins.push(Mat::zeros(rows, cols, CV_32FC1).unwrap().to_mat().unwrap());
    }

    let mut local_orientation_bins: Vec<Mat> = Vec::new();
    let local_bin_size: f32 = 45.0;
    for _ in 0..8 {
        local_orientation_bins.push(Mat::zeros(rows, cols, CV_32FC1)
            .unwrap().to_mat().unwrap());
    }

    for i in 0..rows {
        for j in 0..cols {
            let ixx = get_pixel::<f32>(&ix, i, j);
            let iyy = get_pixel::<f32>(&iy, i, j);
            let mut theta = (iyy / (ixx + 1e-4)).atan() * (180.0 / PI);
            // Convert theta to [0, 360.0]
            if theta < 0.0 {
                theta += 360.0;
            }
            let raw_bin = (theta + 0.5 * bin_size) / bin_size;
            let bin = (raw_bin as i32) % 36;
            set_pixel::<f32>(&mut orientation_bins[bin as usize], i, j, 
                      get_pixel::<f32>(&orientation, i, j));

            let local_raw_bin = (theta + 0.5 * local_bin_size) / local_bin_size;
            let local_bin = (local_raw_bin as i32) % 8;
            set_pixel::<f32>(&mut local_orientation_bins[local_bin as usize], i, j, 
                             get_pixel::<f32>(&orientation, i, j));
        }
    }

    let mut buffer = Mat::default()?;
    for i in 0..36 {
        orientation_bins[i].copy_to(&mut buffer).unwrap();
        gaussian_blur(&buffer, &mut orientation_bins[i], 
                      Size::new(7, 7), 3.0, 0.0, BORDER_DEFAULT).unwrap();
    }
    for i in 0..8 {
        local_orientation_bins[i].copy_to(&mut buffer).unwrap();
        gaussian_blur(&buffer, &mut local_orientation_bins[i],
                      Size::new(7, 7), 3.0, 0.0, BORDER_DEFAULT).unwrap();
    }

    let mut main_orientation = Mat::zeros(rows, cols, CV_8UC1).unwrap().to_mat().unwrap();
    let mut local_main_orientation = Mat::zeros(rows, cols, CV_8UC1).unwrap().to_mat().unwrap();
    for i in 0..rows {
        for j in 0..cols {
            let mut cur_orientation: u8 = 0;
            let mut cur_orientation_value: f32 = 0.0;
            for k in 0..36 {
                let val = get_pixel::<f32>(&orientation_bins[k], i, j);
                if val > cur_orientation_value {
                    cur_orientation_value = val;
                    cur_orientation = k as u8;
                }
            }
            set_pixel::<u8>(&mut main_orientation, i, j, cur_orientation);

            let mut cur_local_orientation: u8 = 0;
            let mut cur_local_orientation_value: f32 = 0.0;
            for k in 0..8 {
                let val = get_pixel::<f32>(&local_orientation_bins[k], i, j);
                if val > cur_local_orientation_value {
                    cur_local_orientation_value = val;
                    cur_local_orientation = k as u8;
                }
            }
            set_pixel::<u8>(&mut local_main_orientation, i, j, cur_local_orientation);
        }
    }

    let feature_num = feature_points.len();
    unsafe {
        feature_mat.create_rows_cols(feature_num as i32, 128, CV_32FC1).unwrap();
    }
    for i in 0..feature_num {
        let x = feature_points[i].x;
        let y = feature_points[i].y;
        let angle: f32 = (get_pixel::<u8>(&main_orientation, y, x) as f32 + 0.5) * bin_size;

        let descriptor_rotate_bin: i32;
        if angle < 45.0 / 2.0 {
            descriptor_rotate_bin = 0;
        } else {
            descriptor_rotate_bin = 1 + (angle - 22.5) as i32 / 45;
        } 

        let p = Point2f::new(x as f32, y as f32);
        let mut rotated_local_main_orientation = Mat::default()?;
        let rotation_matrix = opencv::imgproc::get_rotation_matrix_2d(p, -angle as f64, 1.0).unwrap();
        opencv_utils::warp_affine_with_default(&local_main_orientation, 
                                               &mut rotated_local_main_orientation, 
                                               &rotation_matrix).unwrap();

        let mut out_descriptors: Vec<f32> = Vec::new();
        out_descriptors.reserve(128);

        // Construct a histogram in each of 4x4 grid.
        let st_x = [-8, -4, 0, 4];
        let st_y = [-8, -4, 0, 4];
        for sx in &st_x {
            for sy in &st_y {
                let mut hist: [f32 ;8] = [0.0; 8];

                let xx = sx + x;
                let yy = sy + y;

                for xxx in xx..xx+4 {
                    for yyy in yy..yy+4 {
                        let mut local_main_bin = get_pixel::<u8>(&local_main_orientation, yyy, xxx) as i32;
                        
                        local_main_bin -= descriptor_rotate_bin;
                        local_main_bin = (local_main_bin + 8) % 8;
                        hist[local_main_bin as usize] += 1.0 / 16.0;
                    }
                }

                let mut his_sum = 0.0;
                for ii in 0..8 {
                    if hist[ii] > 0.2 {
                        hist[ii] = 0.2;
                    }
                    his_sum += hist[ii];
                }

                for ii in 0..8 {
                    hist[ii] /= his_sum;
                    out_descriptors.push(hist[ii]);
                }
            }
        }

        for ii in 0..128 {
            set_pixel::<f32>(feature_mat, i as i32, ii, out_descriptors[ii as usize]);
        }
    }

    Ok(())
}
