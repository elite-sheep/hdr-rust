// Copyright 2020 Yuchen Wong

use opencv::core::{ CV_32FC1, Mat, Point, MatTrait, NORM_L2};
use std::error::Error;

pub fn match_feature(feature_mat1: &Mat, 
             feature_mat2: &Mat,
             threshold: f64) -> Result<Vec<Point>, Box<dyn Error>> {
    let feature_num1 = feature_mat1.rows();
    let feature_num2 = feature_mat2.rows();

    let mut feature_match: Vec<Point> = Vec::new();
    let mut match_num: i32 = 0;
    for i in 0..feature_num2 {
        let cur_feature2 = feature_mat2.row(i).unwrap();

        let mut first_match_value = std::f64::MAX;
        let mut first_match: i32 = 0;
        let mut second_match_value = std::f64::MAX;
        let mut _second_match: i32 = 0;

        for j in 0..feature_num1 {
            let cur_feature1 = feature_mat1.row(j).unwrap();
            let norm = opencv::core::norm2(&cur_feature1, &cur_feature2,
                                           NORM_L2, &Mat::default()?).unwrap();
            if norm < first_match_value {
                second_match_value = first_match_value;
                _second_match = first_match;
                first_match_value = norm;
                first_match = j;
            } else if norm < second_match_value {
                second_match_value = norm;
                _second_match = j;
            }
        }

        if first_match_value / second_match_value < threshold {
            match_num += 1;
            feature_match.push(Point::new(first_match, i));
        }
    }

    log::trace!("Number of pairs of features matched: {}.", match_num);

    Ok(feature_match)
}
