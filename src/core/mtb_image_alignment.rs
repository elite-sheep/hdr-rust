/* Copyright 2020 Yuchen Wong */

use opencv::core::{CV_8UC1, Mat, Size};
use opencv::imgproc::{resize, warp_affine};
use opencv::prelude::{MatTrait, Vector};
use opencv::types::VectorOfMat;
use std::error::Error;

#[path = "../base/math_utils.rs"] mod math_utils;
#[path = "../base/opencv_utils.rs"] mod opencv_utils;

pub fn align(images: &VectorOfMat,
             aligned_images: &mut VectorOfMat,
             max_level: u8) 
    -> Result<(), Box<dyn Error>> {

    let pivot: usize = images.len() >> 1;

    let mut pivot_image_pyramid_mtb: VectorOfMat = VectorOfMat::new();
    let mut pivot_image_pyramid_exor: VectorOfMat = VectorOfMat::new();
    compute_image_pyramid(images[pivot], image_pyramid_mtb, image_pyramid_exor, max_level)?;

    let move_x: [i32, 9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    let move_y: [i32, 9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    for i in 0..images.len() {
        if i == pivot {
            aligned_images.push(images[pivot]);
        } else {
            let mut offset_x: i32 = 0;
            let mut offset_y: i32 = 0;
            let mut image_pyramid_mtb: VectorOfMat = VectorOfMat::new();
            let mut image_pyramid_exor: VectorOfMat = VectorOfMat::new();
            for j in 0..max_level {
                offset_x = offset_x << 1;
                offset_y = offset_y << 1;
                for k in 0..9 {
                    let translation_matrix = 
                        math_utils::get_translation_matrix(offset_x + move_x[k], offset_y + move_y[k]);

                    let mut cur_mtb: Mat = Mat::default()?;
                    let mut cur_exor: Mat = Mat::default()?;
                    warp_affine(&image_pyramid_mtb[max_level-j-1], &mut cur_mtb, &translation_matrix);
                    warp_affine(&image_pyramid_exor[max_level-j-1], &mut cur_exor, &translation_matrix);
                }
            }
        }
    }
}

fn compute_image_pyramid(src: &Mat,
                         out_mtb_images: &mut VectorOfMat,
                         out_exclusive_images: &mut VectorOfMat,
                         max_level: u8)
    -> Result<(), Box<dyn Error>> {
        let mut src_clone = src.clone();
        let scale: f32 = 1.0;
        for i in 0..max_level {
            let mut mtb_image = Mat::default()?;
            let mut exclusive_image = Mat::default()?;
            out_mtb_images.push(opencv_utils::compute_mtb_image(&src_clone, &mut mtb_image));
            out_exclusive_images.push(opencv_utils::compute_exclusive_image(&src_clone, &mut exclusive_image, 4));

            resize(&src, &mut src_clone, Size(), scale * 0.5, scale * 0.5);
            scale *= 0.5;
        }

        Ok(())
}
