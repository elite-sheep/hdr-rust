/* Copyright 2020 Yuchen Wong*/

use opencv::core::{Mat, MatExprTrait, Vec3b, Vec3f};
use opencv::prelude::*;
use opencv::types::VectorOfMat;
use std::error::Error;
use std::time::SystemTime;

#[path = "../base/math_utils.rs"] mod math_utils;
#[path = "../base/opencv_utils.rs"] mod opencv_utils;

use math_utils::{hat};
use opencv_utils::matmul;

pub fn solve(images: &VectorOfMat,
             shutter_speeds: &Vec<f32>,
             sample_num: i32,
             lambda: f32,
             out_hdri: &mut Mat) -> Result<(), Box<dyn Error>> {

    log::trace!("Debevec crf_solver started.");

    let start = SystemTime::now();

    // generate weights with a hat function.
    let mut weights: [f32; 256] = [0.0; 256];
    for i in 0..256 {
        weights[i] = hat(i as f32, 0.0, 255.0);
    }

    let rows: i32 = images.get(0)?.rows();
    let cols: i32 = images.get(0)?.cols();

    // shape the output radiance map.
    unsafe {
        out_hdri.create_rows_cols(rows, cols, opencv::core::CV_32FC3)?;
    }

//    let mut samples_x: Vec<i32> = Vec::with_capacity(sample_num as usize);
//    let mut samples_y: Vec<i32> = Vec::with_capacity(sample_num as usize);
//    for _i in 0..sample_num {
//        samples_x.push(math_utils::gen_random_integer(0, cols));
//        samples_y.push(math_utils::gen_random_integer(0, rows));
//    }

    // Solve the radiance map for each channel
    for c in 0..3 {
        solve_internal(images, shutter_speeds, lambda, &weights, sample_num, c as usize, out_hdri)?;
    }

    let duration = start.elapsed()?.as_secs();
    log::trace!("Debevec crf_solver ended: {}.", duration);
    Ok(())
}

fn solve_internal(images: &VectorOfMat,
                  shutter_speeds: &Vec<f32>,
                  lambda: f32,
                  weights: &[f32; 256],
                  sample_number: i32,
                  channel: usize,
                  out_hdri: &mut Mat) -> Result<(), Box<dyn Error>> {

    log::trace!("Solving Debevec CRF for channel {}.", channel);

    let mut samples_x: Vec<i32> = Vec::default();
    let mut samples_y: Vec<i32> = Vec::default();
    histogram_sampling(images, channel, sample_number, 8, &mut samples_x, &mut samples_y)?;

    let rows = images.get(0)?.rows();
    let cols = images.get(0)?.cols();
    let sample_num: usize = samples_x.len();
    let image_num: i32 = images.len() as i32;

    let total_sample_num = (sample_num as i32) * image_num;
    let mut a: Mat = Mat::zeros(
        1+254+total_sample_num, (256+sample_num) as i32, opencv::core::CV_32FC1)?.to_mat()?;
    let mut b: Mat = Mat::zeros(
        (1+254+total_sample_num) as i32, 1, opencv::core::CV_32FC1)?.to_mat()?;

    let mut l = 0;
    for p in 0..image_num {
        let cur_image: Mat = images.get(p as usize)?;
        for i in 0..sample_num {
            // log::trace!("{} {}", samples_y[i], samples_x[i]);
            let z: Vec3b = *cur_image.at_2d::<Vec3b>(samples_x[i], samples_y[i])?;
            *a.at_2d_mut::<f32>(l, z[channel] as i32).unwrap() = 1.0 * weights[z[channel] as usize];
            *a.at_2d_mut::<f32>(l, (256 + i) as i32).unwrap() = -1.0 * weights[z[channel] as usize];
            *b.at_2d_mut::<f32>(l, 0).unwrap() = shutter_speeds[p as usize].ln() * weights[z[channel] as usize];
            l += 1;
        }
    }

    *a.at_2d_mut::<f32>(l, 127).unwrap() = 1.0;
    l += 1;

    for i in 1..255 {
        *a.at_2d_mut::<f32>(l, i-1).unwrap() = lambda * weights[i as usize];
        *a.at_2d_mut::<f32>(l, i).unwrap() = -2.0 * lambda * weights[i as usize];
        *a.at_2d_mut::<f32>(l, i+1).unwrap() = lambda * weights[i as usize];
        l += 1;
    }

    log::trace!("Finishing setting up solver.");

    let mut a_inv: Mat = Mat::default()?;
    opencv::core::invert(&a, &mut a_inv, opencv::core::DECOMP_SVD)?;

    log::trace!("Finishing solving SVD.");

    let mut x: Mat = Mat::default()?;
    matmul(&a_inv, &b, opencv::core::CV_32FC1, &mut x)?;

    let mut g: [f32; 256] = [0.0; 256];
    for i in 0..256 {
        g[i as usize] = *x.at_2d::<f32>(i, 0).unwrap();
    }

    log::trace!("Starting recovering for channel {}.", channel);
    for row in 0..rows {
        for col in 0..cols {
            let mut sum_weight: f32 = 0.0;
            let mut sum_radiance: f32 = 0.0;
            for p in 0..image_num {
                let z: Vec3b = *images.get(p as usize)?.at_2d::<Vec3b>(row, col).unwrap();
                sum_weight += weights[z[channel] as usize];
                sum_radiance += weights[z[channel] as usize] * (g[z[channel] as usize] - shutter_speeds[p as usize].ln());
            }
            out_hdri.at_2d_mut::<Vec3f>(row, col).unwrap()[channel] = (sum_radiance / (sum_weight+0.0001)).exp();
        }
    }

    log::trace!("Finishing solving Debevec CRF for channel {}s.", channel);
    Ok(())
}

fn histogram_sampling(images: &VectorOfMat,
                      channel: usize,
                      sample_num: i32,
                      bin_num: i32,
                      out_sample_x: &mut Vec<i32>,
                      out_sample_y: &mut Vec<i32>) -> Result<(), Box<dyn Error>> {
    let image_num: usize = images.len();
    let rows = images.get(0)?.rows();
    let cols = images.get(0)?.cols();

    let mut bin_x: Vec<Vec<i32>> = Vec::with_capacity(bin_num as usize);
    let mut bin_y: Vec<Vec<i32>> = Vec::with_capacity(bin_num as usize);
    let bin_size: i32 = 256 / bin_num;
    for _i in 0..bin_num {
        bin_x.push(Vec::with_capacity(bin_size as usize));
        bin_y.push(Vec::with_capacity(bin_size as usize));
    }

    for i in 0..rows {
        for j in 0..cols {
            let mut sum_value: i32 = 0;
            for k in 0..image_num {
                sum_value += images.get(k)?.at_2d::<Vec3b>(i, j).unwrap()[channel] as i32;
            }
            sum_value /= image_num as i32;
            bin_x[(sum_value / bin_size) as usize].push(i);
            bin_y[(sum_value / bin_size) as usize].push(j);
        }
    }

    let samples_per_bin: i32 = sample_num / bin_num;
    for i in 0..bin_num {
        let cur_bin_size = bin_x[i as usize].len() as i32;
        for _j in 0..samples_per_bin {
            let cur_index = math_utils::gen_random_integer(0, cur_bin_size);
            out_sample_x.push(bin_x[i as usize][cur_index as usize]);
            out_sample_y.push(bin_y[i as usize][cur_index as usize]);
        }
    }

    log::trace!("Finishing sampling.");
    Ok(())
}
