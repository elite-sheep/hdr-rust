/* Copyright 2020 Yuchen Wong */

use opencv::core::{Mat};
use opencv::prelude::{MatTrait, Vector};
use opencv::types::VectorOfMat;
use std::convert::TryInto;
use std::error::Error;

#[path = "../base/math_utils.rs"] mod math_utils;
#[path = "../base/opencv_utils.rs"] mod opencv_utils;

pub fn align(images: &VectorOfMat,
             aligned_images: &mut VectorOfMat,
             max_level: u8) 
    -> Result<(), Box<dyn Error>> {

    log::trace!("Start MTB Alignment.");
    let pivot: usize = images.len() >> 1;

    log::info!("Align pivot is {}.", pivot);

    let mut pivot_image_pyramid_mtb: VectorOfMat = VectorOfMat::new();
    let mut pivot_image_pyramid_exor: VectorOfMat = VectorOfMat::new();
    compute_image_pyramid(&images.get(pivot)?, &mut pivot_image_pyramid_mtb, &mut pivot_image_pyramid_exor, max_level)?;

    let move_x: [i32; 9] = [-1, -1, -1, 0, 0, 0, 1, 1, 1];
    let move_y: [i32; 9] = [0, -1, 1, -1, 0, 1, 1, 0, -1];
    for i in 0..images.len() {
        if i == pivot {
            aligned_images.push(images.get(pivot)?.clone()?);
            continue;
        } else {
            let mut offset_x: i32 = 0;
            let mut offset_y: i32 = 0;
            let mut image_pyramid_mtb: VectorOfMat = VectorOfMat::new();
            let mut image_pyramid_exor: VectorOfMat = VectorOfMat::new();
            compute_image_pyramid(&images.get(i)?, &mut image_pyramid_mtb, &mut image_pyramid_exor, max_level)?;
            for j in 0..max_level {
                offset_x = offset_x * 2;
                offset_y = offset_y * 2;

                let mut best_move: usize = 0;
                let mut best_similarity: f64 = -1.0;
                for k in 0..9 {
                    let mut translation_matrix: Mat = Mat::default()?;
                    math_utils::get_translation_matrix(&mut translation_matrix, offset_x + move_x[k], offset_y + move_y[k])?;

                    let mut cur_mtb: Mat = Mat::default()?;
                    let mut cur_exor: Mat = Mat::default()?;
                    opencv_utils::warp_affine_with_default(
                        &image_pyramid_mtb.get((max_level-j-1).try_into().unwrap())?, &mut cur_mtb, &translation_matrix)?;
                    opencv_utils::warp_affine_with_default(
                        &image_pyramid_exor.get((max_level-j-1).try_into().unwrap())?, &mut cur_exor, &translation_matrix)?;

                    let cur_similarity: f64 = compute_image_similarity(&cur_mtb, &cur_exor, 
                                                    &pivot_image_pyramid_mtb.get((max_level-j-1).try_into().unwrap())?, 
                                                    &pivot_image_pyramid_exor.get((max_level-j-1).try_into().unwrap())?)?;
                    // We tend not to move the image, so we use the moving distance as
                    // a penalty as this stage.
                    let penalty = (move_x[k].abs() as f64) + (move_y[k].abs() as f64);

                    if best_similarity < 0.0 || best_similarity > cur_similarity + penalty {
                        best_similarity = cur_similarity + penalty;
                        best_move = k;
                    }
                }

                offset_x += move_x[best_move];
                offset_y += move_y[best_move];
            }

            log::info!("Image {} with offset {}, {}.", i, offset_x, offset_y);

            // Do the final translation
            let mut final_translation_image: Mat = Mat::default()?;
            let mut translation_matrix: Mat = Mat::default()?;
            math_utils::get_translation_matrix(&mut translation_matrix, offset_x, offset_y)?;
            opencv_utils::warp_affine_with_default(
                &images.get(i)?, &mut final_translation_image, &translation_matrix)?;
            aligned_images.push(final_translation_image);
        }
    }

    log::trace!("MTB Alignment finished.");
    Ok(())
}

fn compute_image_pyramid(src: &Mat,
                         out_mtb_images: &mut VectorOfMat,
                         out_exclusive_images: &mut VectorOfMat,
                         max_level: u8)
    -> Result<(), Box<dyn Error>> {
        let mut src_clone: Mat = src.clone()?;
        let mut scale: f64 = 1.0;
        for _i in 0..max_level {
            let mut mtb_image: Mat = Mat::default()?;
            let mut exclusive_image: Mat = Mat::default()?;
            log::trace!("Resizing with scale {}.", scale);
            opencv_utils::compute_mtb_image(&src_clone, &mut mtb_image)?;
            opencv_utils::compute_exclusive_image(&src_clone, &mut exclusive_image, 4)?;

            out_mtb_images.push(mtb_image);
            out_exclusive_images.push(exclusive_image);

            scale *= 0.5;
            opencv_utils::resize_image_with_default(src, &mut src_clone, scale, scale)?;
        }

        Ok(())
}

fn compute_image_similarity(a_mtb: &Mat,
                            a_exor: &Mat,
                            b_mtb: &Mat,
                            b_exor: &Mat)
    -> Result<f64, Box<dyn Error>> {
    let mut image_and = Mat::default()?;
    let mut image_xor = Mat::default()?;
    opencv::core::bitwise_xor(a_mtb, b_mtb, &mut image_xor, &opencv::core::no_array()?)?;
    opencv::core::bitwise_and(&image_xor, b_exor, &mut image_and, &opencv::core::no_array()?)?;
    opencv::core::bitwise_and(&image_and, a_exor, &mut image_xor, &opencv::core::no_array()?)?;
    let sum: f64 = opencv::core::sum_elems(&image_xor).unwrap()[0];

    log::info!("Computed image similarity: {}", sum);

    Ok(sum / 255.0)
}
