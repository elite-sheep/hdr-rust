/* Copyright 2020 Yuchen Wong*/

use opencv::core::{Mat, MatExpr, MatExprTrait, Matx_MulOp, Vec3b, Vec3f};
//use opencv::prelude::{MatTrait, Vector};
use opencv::prelude::*;
use opencv::types::VectorOfMat;
use std::error::Error;

#[path = "../base/math_utils.rs"] mod math_utils;
#[path = "../base/opencv_utils.rs"] mod opencv_utils;

pub fn solve(images: &VectorOfMat,
             shutter_speeds: &Vec<f32>,
             lambda: f32,
             out_hdri: &mut Mat) -> Result<(), Box<dyn Error>> {

    log::trace!("Debevec crf_solver started.");

    // generate weights with a hat function.
    let mut weights: [f32; 256] = [0.0; 256];
    for i in 0..256 {
        weights[i] = math_utils::hat(i as f32, 0.0, 255.0);
    }

    let image_num = images.len();
    let rows: i32 = images.get(0)?.rows();
    let cols: i32 = images.get(0)?.cols();

    // shape the output radiance map.
    unsafe {
        out_hdri.create_rows_cols(rows, cols, opencv::core::CV_32FC3)?;
    }

    // generate samples randomly.
    let sample_num = 100;
    let mut samples_x: Vec<i32> = Vec::with_capacity(sample_num as usize);
    let mut samples_y: Vec<i32> = Vec::with_capacity(sample_num as usize);
    for _i in 0..sample_num {
        samples_x.push(math_utils::gen_random_integer(0, cols));
        samples_y.push(math_utils::gen_random_integer(0, rows));
    }

    // Solve the radiance map for each channel
    for c in 0..3 {
        solve_internal(images, shutter_speeds, lambda, &weights, &samples_x, &samples_y, c, out_hdri)?;
    }

    log::trace!("Debevec crf_solver ended..");
    Ok(())
}

fn solve_internal(images: &VectorOfMat,
                  shutter_speeds: &Vec<f32>,
                  lambda: f32,
                  weights: &[f32; 256],
                  samples_x: &Vec<i32>,
                  samples_y: &Vec<i32>,
                  channel: i32,
                  out_hdri: &mut Mat) -> Result<(), Box<dyn Error>> {

    log::trace!("Solving Debevec CRF for channel {}.", channel);

    let rows = images.get(0)?.rows();
    let cols = images.get(0)?.cols();
    let sample_num: usize = samples_x.len();
    let image_num: i32 = images.len() as i32;

    let total_sample_num = (sample_num as i32) * image_num;
    let mut A: Mat = Mat::zeros(
        1+254+total_sample_num, (256+sample_num) as i32, opencv::core::CV_32FC1)?.to_mat()?;
    let mut B: Mat = Mat::zeros(
        (1+254+total_sample_num) as i32, 1, opencv::core::CV_32FC1)?.to_mat()?;

    let mut l = 0;
    for p in 0..image_num {
        let cur_image: Mat = images.get(p as usize)?;
        for i in 0..sample_num {
            // log::trace!("{} {}", samples_y[i], samples_x[i]);
            let z: Vec3b = *cur_image.at_2d::<Vec3b>(samples_y[i], samples_x[i])?;
            *A.at_2d_mut::<f32>(l, z[channel as usize] as i32).unwrap() = 1.0 * weights[z[channel as usize] as usize];
            *A.at_2d_mut::<f32>(l, (256 + i) as i32).unwrap() = -1.0 * weights[z[channel as usize] as usize];
            *B.at_2d_mut::<f32>(l, 0).unwrap() = shutter_speeds[p as usize].ln() * weights[z[channel as usize] as usize];
            l += 1;
        }
    }

    *A.at_2d_mut::<f32>(l, 127).unwrap() = 1.0;
    l += 1;

    for i in 1..255 {
        *A.at_2d_mut::<f32>(l, i-1).unwrap() = lambda * weights[i as usize];
        *A.at_2d_mut::<f32>(l, i).unwrap() = -2.0 * lambda * weights[i as usize];
        *A.at_2d_mut::<f32>(l, i+1).unwrap() = lambda * weights[i as usize];
        l += 1;
    }

    log::trace!("Finishing setting up solver.");

    let mut A_inv: Mat = Mat::default()?;
    opencv::core::invert(&A, &mut A_inv, opencv::core::DECOMP_SVD)?;

    log::trace!("Finishing solving SVD.");
    log::trace!("{} {} {} {} {} {}", A_inv.rows(), A_inv.cols(), A.channels()?, B.rows(), B.cols(), B.channels()?);

    let mut X: Mat = Mat::default()?;
    opencv_utils::matmul(&A_inv, &B, opencv::core::CV_32FC1, &mut X)?;

    let mut g: [f32; 256] = [0.0; 256];
    for i in 0..256 {
        g[i as usize] = *X.at_2d::<f32>(i, 0).unwrap();
    }

    log::trace!("Starting recovering for channel {}.", channel);
    for row in 0..rows {
        for col in 0..cols {
            let mut sum_weight: f32 = 0.0;
            let mut sum_radiance: f32 = 0.0;
            for p in 0..image_num {
                let z: Vec3b = *images.get(p as usize)?.at_2d::<Vec3b>(row, col).unwrap();
                sum_weight += weights[z[channel as usize] as usize];
                sum_radiance += weights[z[channel as usize] as usize] * (g[z[channel as usize] as usize] - shutter_speeds[p as usize].ln());
            }
            out_hdri.at_2d_mut::<Vec3f>(row, col).unwrap()[channel as usize] = (sum_radiance / sum_weight).exp();
        }
    }

    log::trace!("Finishing solving Debevec CRF for channel {}.", channel);
    Ok(())
}