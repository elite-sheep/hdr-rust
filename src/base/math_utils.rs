/* Copyright 2020 Yuchen Wong */

use opencv::core::{Mat}

// This file contains some helper functions of math
// It is designed and implemented following c coding style.

pub fn get_translation_matrix(tx: i32,
                              ty: i32)
    -> Mat {
    let res_mat: Mat = Mat::zeros(2, 3, CV_32FC1);
    *res_mat.at_2d_mut::<f32>(0, 0).unwrap() = 1;
    *res_mat.at_2d_mut::<f32>(1, 1).unwrap() = 1;
    *res_mat.at_2d_mut::<f32>(0, 2).unwrap() = tx;
    *res_mat.at_2d_mut::<f32>(1, 2).unwrap() = ty;

    return res_mat;
}
