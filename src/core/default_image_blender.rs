// Copyright 2020 Yuchen Wong

use opencv::core::{ Mat, MatTrait, Point, Vec3b, Vec3f, CV_8UC1, CV_8UC3, CV_32FC3 };
use opencv::core::prelude::{ MatExprTrait };
use std::error::Error;

#[path="../base/opencv_utils.rs"] mod opencv_utils;

use opencv_utils::{ get_pixel, set_pixel };

pub fn blend_image(images: &Vec<Mat>,
                   alignments: &Vec<Point>,
                   out_panorama: &mut Mat) -> Result<(), Box<dyn Error>> {
    log::trace!("Image blending: Start.");

    let image_num = images.len();
    let mut accumulated_offset: Vec<Point> = Vec::new();

    let mut min_dy: i32 = 0;
    let mut max_dy: i32 = 0;
    for i in 0..image_num-1 {
        if i == 0 {
            accumulated_offset.push(alignments[0]);
        } else {
            accumulated_offset.push(alignments[i]+accumulated_offset[i-1]);
        }

        log::trace!("{}", alignments[i].x);
        if accumulated_offset[i].y < min_dy {
            min_dy = accumulated_offset[i].y;
        }
        if accumulated_offset[i].y > max_dy {
            max_dy = accumulated_offset[i].y;
        }
    }

    // Calculate width and height of panorama
    let mut all_width: i32 = 0;
    let mut all_height: i32 = 0;

    for image in images {
        all_width += image.cols();
    }
    all_width += accumulated_offset[image_num-2].x;

    all_height += images[0].rows();
    all_height -= min_dy;
    all_height += max_dy;

    let mut panorama = Mat::zeros(all_height, all_width, CV_32FC3).unwrap().to_mat().unwrap();
    let mut panorama_weight = Mat::zeros(all_height, all_width, CV_8UC1).unwrap().to_mat().unwrap();
    let mut accumulate_width: i32 = 0;
    for i in 0..image_num {
        let cur_image = &images[i];
        let cur_rows = images[i].rows();
        let cur_cols = images[i].cols();

        let mut start_x = 0;
        if i > 0 {
            start_x = accumulate_width - accumulated_offset[i-1].x;
        }
        let mut start_y = -min_dy;
        if i > 0 {
            start_y += accumulated_offset[i-1].y;
        }

        accumulate_width += cur_image.cols();

        for sr in 0..cur_rows {
            for sc in 0..cur_cols {
                let cur_status = get_pixel::<u8>(&panorama_weight, start_y+sr, start_x+sc);
                let cur_pixel = get_pixel::<Vec3b>(&cur_image, sr, sc);
                if cur_status == 0 {
                    set_pixel::<u8>(&mut panorama_weight, start_y+sr, start_x+sc, 1);
                    let mut pixel = Vec3f::all(0.0);
                    pixel[0] += cur_pixel[0] as f32;
                    pixel[1] += cur_pixel[1] as f32;
                    pixel[2] += cur_pixel[2] as f32;
                    set_pixel::<Vec3f>(&mut panorama, start_y+sr, start_x+sc, pixel);
                } else {
                    let panorama_pixel = get_pixel::<Vec3f>(&panorama, start_y+sr, start_x+sc);
                    let mut pixel = Vec3f::all(0.0);
                    pixel[0] += cur_pixel[0] as f32 * 0.5 + panorama_pixel[0] * 0.5;
                    pixel[1] += cur_pixel[1] as f32 * 0.5 + panorama_pixel[1] * 0.5;
                    pixel[2] += cur_pixel[2] as f32 * 0.5 + panorama_pixel[2] * 0.5;
                    set_pixel::<Vec3f>(&mut panorama, start_y+sr, start_x+sc, pixel);
                }
            }
        }
    }

    panorama.convert_to(out_panorama, CV_8UC3, 1.0, 0.0).unwrap();

    Ok(())
}
