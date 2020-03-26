/* Copyright 2020 Yuchen Wong */

use opencv::core::{CV_8UC1, Mat, Vec3b};
use opencv::prelude::MatTrait;
use std::error::Error;
use std::convert::TryInto;

// This file contains some helper function of opencv
// It is designed and implemented following c coding style.

pub fn compute_mtb_image(src: &Mat, 
                         dst: &mut Mat,
                         low_threshold: f32,
                         high_threshold: f32) 
    -> Result<(), Box<dyn Error>> {
    log::trace!("Compute MTB image starts: low_threshold: {}, high_threshold: {}"
                , low_threshold, high_threshold);

    let rows: i32 = src.rows();
    let cols: i32 = src.cols();
    let mut mtb_pixels: Vec<u8> = Vec::new();

    unsafe {
        mtb_pixels.reserve((rows * cols).try_into().unwrap());
        dst.create_rows_cols(rows, cols, CV_8UC1)?;

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

        let mut mtb_pixels_sorted: Vec<u8> = mtb_pixels.clone();
        mtb_pixels_sorted.sort();

        let mut will_abandon_pixel: bool = false;
        if low_threshold < 0.0 || high_threshold < 0.0 {
            will_abandon_pixel = true;
        }
        let mut low_index: usize = ((rows * cols) >> 1).try_into().unwrap();
        let mut high_index: usize = low_index;
        if !will_abandon_pixel {
            low_index = (((rows * cols) as f32) * low_threshold) as usize;
            high_index = (((rows * cols) as f32) * high_threshold) as usize;
        }
        
        for row in 0..rows {
            for col in 0..cols {
                let index: usize = (row * cols + col).try_into().unwrap();
                let mut mtb_pixel_value = mtb_pixels[index];
                if mtb_pixel_value > mtb_pixels_sorted[high_index] {
                    mtb_pixel_value = 255;
                } else if mtb_pixel_value <= mtb_pixels_sorted[low_index] {
                    mtb_pixel_value = 0;
                }

                let mtb_pixel: &mut u8 = dst.at_2d_mut(row, col)?;
                *mtb_pixel = mtb_pixel_value;
            }
        }
    }

    log::trace!("Compute MTB image ends.");
    Ok(())
}
