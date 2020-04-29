// Copyright 2020 Yuchen Wong

use opencv::core::{ Mat, MatTrait, Point };
use rand::Rng;
use std::cmp;
use std::error::Error;

pub fn match_image(image1: &Mat,
                   _image2: &Mat,
                   features1: &Vec<Point>,
                   features2: &Vec<Point>,
                   feature_matches: &Vec<Point>) -> Result<Point, Box<dyn Error>> {
    let matches_number = feature_matches.len();
    let k = cmp::max(500, 4 * matches_number as i32);

    let mut rng = rand::thread_rng();
    let offset = Point::new(image1.cols(), 0);
    let mut alignment_diff = std::f32::MAX;
    let mut alignment: Point = Point::new(0, 0);
    for _ in 0..k {
        let cur_match: usize = rng.gen_range(0, matches_number);

        let point1 = features1[feature_matches[cur_match].x as usize];
        let point2 = features2[feature_matches[cur_match].y as usize];

        let cur_alignment = point1 - point2 + offset;

        let mut cur_difference = 0.0;
        for match_p in feature_matches {
            let pp1 = features1[match_p.x as usize];
            let pp2 = features2[match_p.y as usize];
            let moved_p = pp2 + offset + cur_alignment;

            let dis_p = moved_p - pp1;
            let diff = (dis_p.x * dis_p.x + dis_p.y * dis_p.y) as f32;

            //if diff < (image1.cols() * image1.cols()) as f32 {
            cur_difference += diff;
            //}
        }

        if cur_difference < alignment_diff {
            alignment_diff = cur_difference;
            alignment = cur_alignment - offset;
            alignment = alignment - offset;
        }
    }

    Ok(alignment)
}
