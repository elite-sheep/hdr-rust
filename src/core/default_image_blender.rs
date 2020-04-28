// Copyright 2020 Yuchen Wong

use opencv::core::{ Mat, MatTrait, Point, CV_32FC1, CV_32FC3 };
use std::error::Error;

pub fn blend_image(images: &Vec<Mat>,
                   alignments: &Vec<Points>,
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

    for image in &images {
        all_width += image.cols();
    }
    all_width += accumulated_offset[image_num-2];

    all_height += images[0].rows();
    all_height -= min_dy;
    all_height += max_dy;

    let mut panorama = Mat::zeros(all_width, all_height, CV_32FC3).unwrap().to_mat().unwrap();
    let mut panorama_weight = Mat::zeros(all_width, all_height, CV_32FC1).unwrap().to_mat().unwrap();
    for i in 0..image_num {
        let cur_image = images[i];
        let cur_rows = images[i].rows();
        let cur_cols = images[i].cols();
    }
}
