// Copyright 2020 Yuchen Wong

use opencv::core::{ CV_8UC3, CV_32FC3, Mat, Point, Vec3f, MatTrait};
use std::error::Error;

#[path = "../base/opencv_utils.rs"] mod opencv_utils;

use opencv_utils::{ get_pixel, set_pixel };

pub fn cylindrial_wrap(src: &Mat,
                       focal_length: f32,
                       dst: &mut Mat) -> Result<(), Box<dyn Error>> {

    let mut float_src: Mat = Mat::default()?;
    src.convert_to(&mut float_src, CV_32FC3, 1.0, 0.0).unwrap();

    let rows = src.rows();
    let cols = src.cols();

    let f = focal_length;
    let invS = 1.0 / f;

    // make center of image o as origin
    let origin_x = rows >> 1;
    let origin_y = cols >> 1;

    let origin_x_max = (rows - 1 - origin_x) as f32;

    let cy_x_max = focal_length * (origin_x_max / focal_length).atan();
    // Cut black edges of out wrapped image.
    let offset = origin_x_max - cy_x_max;

    let mut tmp_wrapped: Mat = Mat::default()?;
    unsafe {
        tmp_wrapped.create_rows_cols(2*cy_x_max as i32, cols, CV_32FC3).unwrap();
    }

    for i in 0..tmp_wrapped.rows() {
        for j in 0..tmp_wrapped.cols() {
            let cy_x: f32 = (i - origin_x) as f32 + offset;
            let cy_y: f32 = (j - origin_y) as f32;

            let mut x: f32 = (cy_x * invS).tan() * f;
            let mut y: f32 = cy_y * (x*x + f*f).sqrt() * invS;

            x += origin_x as f32;
            y += origin_y as f32;

            // If out of bound, then just continue.
            if x < 0.0 || x.ceil() >= rows as f32 || y < 0.0 || y.ceil() >= cols as f32 {
                continue;
            }

            let x_ceil = x.ceil();
            let y_ceil = y.ceil();
            let x_floor = x.floor();
            let y_floor = y.floor() ;

            let xt = x - x_floor;
            let yt = y - y_floor;

            let pixel00 = get_pixel::<Vec3f>(&float_src, x_ceil as i32, y_floor as i32);
            let pixel01 = get_pixel::<Vec3f>(&float_src, x_ceil as i32, y_ceil as i32);
            let pixel02 = get_pixel::<Vec3f>(&float_src, x_floor as i32, y_floor as i32);
            let pixel03 = get_pixel::<Vec3f>(&float_src, x_floor as i32, y_ceil as i32);
            let mut pixel: Vec3f = Vec3f::all(0.0);
            for k in 0..3 {
                pixel[k] = xt * yt * pixel02[k] + xt * (1.0 - yt) * pixel03[k] +
                    (1.0 - xt) * (1.0 - yt) * pixel01[k] + (1.0 - xt) * yt * pixel00[k];
            }

            set_pixel::<Vec3f>(&mut tmp_wrapped, i, j, pixel);
        }
    }

    tmp_wrapped.convert_to(dst, CV_8UC3, 1.0, 0.0).unwrap();

    Ok(())
}
