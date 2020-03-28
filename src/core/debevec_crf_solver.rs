/* Copyright 2020 Yuchen Wong*/

use opencv::core::{Mat}
use opencv::prelude::{MatTrait, Vector};
use opencv::types::VectorOfMat;

#[path = "../base/math_utils.rs"] mod math_utils;

pub fn solve(images: &VectorOfMat,
             shutter_speeds: Vec<f32>,
             out_hdri: &mut Mat) -> Result<(), Box<dyn Error>> {
    // generate weights with a hat function.
    let mut weights: [f32; 256] = [0.0; 256];
    for i in 0..256 {
        weights[i] = math_utils::hat(i as f32, 0, 255);
    }

    let image_num = images.len();
    let rows: i32 = images.get(0)?.rows();
    let cols: i32 = images.get(0)?.cols();

    // shape the output radiance map.
    unsafe {
        out_hdri.new_rows_cols(rows, cols, opencv::core::CV_32FC3);
    }

    // generate samples randomly.
    let sample_num = rows * cols / image_num;
    let samples_x: Vec<i32> = Vec::with_capacity(sample_num as usize);
    let samples_y: Vec<i32> = Vec::with_capacity(sample_num as usize);
    for i in 0..sample_num {
        samples_x[i] = math_utils::gen_random_integer(0, cols);
        samples_y[i] = math_utils::gen_random_integer(0, rows);
    }

    // Solve the radiance map for each channel
    for c in 0..3 {
        solve_internal(images, shutter_speeds, &samples_x, &samples_y, c, out_hdri)?;
    }

    Ok(())
}

fn solve_internal(images: &VectorOfMat,
                  shutter_speeds: &Vec<f32>,
                  samples_x: &Vec<i32>,
                  samples_y: &Vec<i32>,
                  channel: i32,
                  out_hdri: &mut Mat) -> Result<(), Box<dyn Error>> {
    let rows = images.get(0)?.rows();
    let cols = images.get(0)?.cols();
    let sample_num: usize = samples_x.len();

    let mut A: Mat = Mat::zeros(1+254+sample_num, 256+sample_num, CV_32FC1)?.as_mat();
    let mut B: Mat = Mat::zeros(256+sample_num, 1, CV_32FC1)?.as_mat();
}
