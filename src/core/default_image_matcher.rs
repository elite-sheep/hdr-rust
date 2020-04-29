// Copyright 2020 Yuchen Wong

use opencv::core::{ Mat, MatTrait, Point, Scalar };
use rand::Rng;
use std::cmp;
use std::error::Error;

pub fn match_image(image1: &mut Mat,
                   image2: &mut Mat,
                   features1: &Vec<Point>,
                   features2: &Vec<Point>,
                   feature_matches: &Vec<Point>) -> Result<Point, Box<dyn Error>> {
    let matches_number = feature_matches.len();
    let k = cmp::max(500, 4 * matches_number as i32);

    let mut rng = rand::thread_rng();
    let offset = Point::new(image1.cols(), 0);
    let mut alignment_diff = std::f32::MAX;
    let mut alignment: Point = Point::new(0, 0);
    let mut feature_match = Point::new(0, 0);
    for _ in 0..k {
        let cur_match: usize = rng.gen_range(0, matches_number);

        let point1 = features1[feature_matches[cur_match].x as usize];
        let point2 = features2[feature_matches[cur_match].y as usize];

        let cur_alignment = point2 - point1 + offset;
        if (cur_alignment.x * cur_alignment.x + cur_alignment.y * cur_alignment.y) as f32 >
            (image1.cols() * image1.cols()) as f32 {
                continue;
        }

        let mut cur_difference = 0.0;
        for match_p in feature_matches {
            let pp1 = features1[match_p.x as usize];
            let pp2 = features2[match_p.y as usize];
            let moved_p = pp2 + offset + cur_alignment;

            let dis_p = moved_p - pp1;
            let diff = (dis_p.x * dis_p.x + dis_p.y * dis_p.y) as f32;

            if diff < (image1.cols() * image1.cols()) as f32 {
                cur_difference += diff;
            }
        }

        if cur_difference < alignment_diff {
            alignment_diff = cur_difference;
            alignment = cur_alignment;
            feature_match = feature_matches[cur_match];
            //alignment = alignment - offset;
        }
    }
    opencv::imgproc::circle(image1, features1[feature_match.x as usize], 5, Scalar::new(0.0, 255.0, 0.0, 1.0), 1, 8, 0).unwrap();
    opencv::imgproc::circle(image2, features2[feature_match.y as usize], 5, Scalar::new(0.0, 255.0, 0.0, 1.0), 1, 8, 0).unwrap();

    Ok(alignment)
}
