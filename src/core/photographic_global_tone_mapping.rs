/* Copyright 2020 Yuchen Wong */

use opencv::core::{Mat, Scalar_, Vec3f};
use opencv::prelude::{MatTrait, MatExprTrait, Vector};
use opencv::types::{VectorOfMat};
use std::error::Error;

#[path = "../base/math_utils.rs"] mod math_utils;

pub fn map(src: &Mat,
           a: f32,
           l_white: f32,
           out_ldr: &mut Mat) -> Result<(), Box<dyn Error>> {
    log::trace!("Starting tone mapping: PhotoGraphics Global.");

    let mut l_w: Mat = Mat::default()?;
    let mut radiance_map: Mat = Mat::default()?;
    compute_radiance(src, a, l_white, &mut l_w, &mut radiance_map)?;

    let mut tmp_ldr: Mat = src.clone()?;
    let mut tmp_ldr_array: VectorOfMat = VectorOfMat::new();
    let mut out_ldr_array: VectorOfMat = VectorOfMat::new();
    opencv::core::split(&tmp_ldr, &mut tmp_ldr_array).unwrap();
    for i in 0..3 {
        let cur_mat: Mat = tmp_ldr_array.get(i).unwrap();
        log::info!("{} {} {} {} {} {}", cur_mat.rows(), cur_mat.cols(), cur_mat.channels().unwrap(), radiance_map.rows(), radiance_map.cols(), radiance_map.channels().unwrap());
        let mut tmp_mat: Mat = Mat::default()?;
        let mut out_mat: Mat = Mat::default()?;
        opencv::core::divide2(&cur_mat, &l_w, &mut tmp_mat, 1.0, opencv::core::CV_32FC1).unwrap();
        opencv::core::multiply(&tmp_mat, &radiance_map, &mut out_mat, 1.0, opencv::core::CV_32FC1).unwrap();
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
                    a: f32,
                    l_white: f32,
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

    let l_white_inv_2: f32 = 1.0 / (l_white * l_white);
    for i in 0..rows {
        for j in 0..cols {
            let cur_l_w: f32 = *out_l_w.at_2d::<f32>(i, j).unwrap();
            let cur_l: f32 = (cur_l_w / l_w_hat) * a;
            let cur_l_d: f32 = (cur_l * (1.0 + cur_l * l_white_inv_2)) / (1.0 + cur_l);
            *out_radiance_map.at_2d_mut::<f32>(i, j).unwrap() = cur_l_d;
        }
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
