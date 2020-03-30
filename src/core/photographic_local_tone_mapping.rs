/* Copyright 2020 Yuchen Wong */

use opencv::core::{Mat, Scalar_, Size_, Vec3f};
use opencv::prelude::{MatTrait, MatExprTrait, Vector};
use opencv::types::{VectorOfMat};
use std::error::Error;

#[path = "../base/math_utils.rs"] mod math_utils;

pub fn map(src: &Mat,
           alpha: f32,
           phi: f32,
           epsilon: f32,
           max_kernel_size: i32,
           out_ldr: &mut Mat) -> Result<(), Box<dyn Error>> {
    log::trace!("Starting tone mapping: PhotoGraphics Local.");

    let mut l_d: Mat = Mat::default()?;
    let mut l_w: Mat = Mat::default()?;
    compute_radiance(src, alpha, phi, epsilon, max_kernel_size, &mut l_w, &mut l_d).unwrap();

    let mut tmp_ldr: Mat = src.clone()?;
    let mut tmp_ldr_array: VectorOfMat = VectorOfMat::new();
    let mut out_ldr_array: VectorOfMat = VectorOfMat::new();
    opencv::core::split(&tmp_ldr, &mut tmp_ldr_array).unwrap();
    for i in 0..3 {
        let cur_mat: Mat = tmp_ldr_array.get(i).unwrap();
        let mut tmp_mat: Mat = Mat::default()?;
        let mut out_mat: Mat = Mat::default()?;
        opencv::core::divide2(&cur_mat, &l_w, &mut tmp_mat, 1.0, opencv::core::CV_32FC1).unwrap();
        opencv::core::multiply(&tmp_mat, &l_d, &mut out_mat, 1.0, opencv::core::CV_32FC1).unwrap();
        out_ldr_array.push(out_mat);
    }
    opencv::core::merge(&out_ldr_array, &mut tmp_ldr).unwrap();

    let mut ldr_uncropped: Mat = Mat::default()?;
    opencv::core::multiply(&tmp_ldr, &Scalar_::all(255.0), &mut ldr_uncropped, 1.0, -1).unwrap();
    ldr_uncropped.convert_to(out_ldr, opencv::core::CV_8UC3, 1.0, 0.0).unwrap();

    log::trace!("Tone mapping finished: PhotoGraphics Global.");
    Ok(())
}

fn compute_radiance(src: &Mat,
                    alpha: f32,
                    phi: f32,
                    epsilon: f32,
                    max_kernel_size: i32,
                    out_l_w: &mut Mat,
                    out_radiance_map: &mut Mat) -> Result<(), Box<dyn Error>> {
    compute_l_w(src, out_l_w).unwrap();

    let mut l_w_log = Mat::default()?;
    let mut l_w_tmp: Mat = Mat::default()?;
    opencv::core::add(out_l_w, &Scalar_::all(0.0001), 
                      &mut l_w_tmp, &opencv::core::no_array()?, opencv::core::CV_32FC1).unwrap();
    opencv::core::log(&l_w_tmp, &mut l_w_log).unwrap();
    l_w_tmp.release()?;

    let l_w_hat: f32 = opencv::core::mean(&l_w_log, &opencv::core::no_array()?).unwrap()[0].exp() as f32;
    log::trace!("{}", l_w_hat);

    let rows: i32 = src.rows();
    let cols: i32 = src.cols();
    unsafe {
        out_radiance_map.create_rows_cols(rows, cols, opencv::core::CV_32FC1).unwrap();
    }

    let mut l_m: Mat = Mat::default()?;
    let mut l_m_tmp: Mat = Mat::default()?;
    opencv::core::divide2(out_l_w, &Scalar_::all(l_w_hat as f64), &mut l_m_tmp, 1.0, opencv::core::CV_32FC1)?;
    opencv::core::multiply(&l_m_tmp, &Scalar_::all(alpha as f64), &mut l_m, 1.0, opencv::core::CV_32FC1)?;

    let mut gaussian_filters: VectorOfMat = VectorOfMat::new();
    let mut cur_size: i32 = 1;
    while cur_size <= max_kernel_size {
        let mut cur_gaussian: Mat = Mat::default()?;
        opencv::imgproc::gaussian_blur(&l_m, &mut cur_gaussian, 
                Size_::new(cur_size, cur_size), 0.0, 0.0, opencv::core::BORDER_DEFAULT)?;
        gaussian_filters.push(cur_gaussian);
        cur_size += 2;
    }

    let mut l_d_down = Mat::default()?;
    unsafe {
        l_d_down.create_rows_cols(rows, cols, opencv::core::CV_32FC1).unwrap();
    }

    log::info!("Starting doing local ops.");
    let gaussian_num = gaussian_filters.len();
    for i in 0..(gaussian_num-1) {
        let s: f32 = 2.0 * (i as f32) + 1.0;
        let cur_gaussian: Mat = gaussian_filters.get(i).unwrap();
        let next_gaussian: Mat = gaussian_filters.get(i+1).unwrap();

        let mut down: Mat = Mat::default()?;
        opencv::core::add(&cur_gaussian, &Scalar_::all((phi.exp2()*alpha/s.powi(2)) as f64), 
                        &mut down, &opencv::core::no_array()?, opencv::core::CV_32FC1)?;
        let mut up: Mat = Mat::default()?;
        opencv::core::subtract(&cur_gaussian, &next_gaussian, &mut up, &opencv::core::no_array()?, opencv::core::CV_32FC1)?;
        let mut v: Mat = Mat::default()?;
        opencv::core::divide2(&up, &down, &mut v, 1.0, opencv::core::CV_32FC1)?;

        for row in 0..rows {
            for col in 0..cols {
                let cur_v: f32 = *v.at_2d::<f32>(row, col).unwrap();
                if cur_v.abs() < epsilon {
                    *l_d_down.at_2d_mut::<f32>(row, col).unwrap() = 1.0 +
                        *cur_gaussian.at_2d::<f32>(row, col).unwrap();
                }
            }
        }
    }

    opencv::core::divide2(&l_m, &l_d_down, out_radiance_map, 1.0, opencv::core::CV_32FC1)?;

    for i in 0..rows {
        let mut test: f64 = 0.0;
        for j in 0..cols {
            test += *l_d_down.at_2d::<f32>(i, j).unwrap() as f64;
        }
        log::info!("{}", test);
    }

    Ok(())
}

fn compute_l_w(src: &Mat,
               dst: &mut Mat) -> Result<(), Box<dyn Error>> {
    let rows = src.rows();
    let cols = src.cols();

    unsafe {
        dst.create_rows_cols(rows, cols, opencv::core::CV_32FC1)?;
    }

    for i in 0..rows {
        for j in 0..cols {
            let pixel_value: Vec3f = *src.at_2d::<Vec3f>(i, j).unwrap();
            let pixel_b = pixel_value[0] as f32;
            let pixel_g = pixel_value[1] as f32;
            let pixel_r = pixel_value[2] as f32;
            *dst.at_2d_mut::<f32>(i, j).unwrap() = 0.06 * pixel_b + 0.67 * pixel_g + 0.27 * pixel_r;
        }
    }

    Ok(())
}
