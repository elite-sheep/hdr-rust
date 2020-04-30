// Copyright 2020 Yuchen Wong

use opencv::core::{ CV_8UC3, CV_32FC1, Mat, Point, Scalar, Size, Vec3b, BORDER_DEFAULT, NORM_MINMAX };
use opencv::imgproc::{ gaussian_blur, spatial_gradient, COLOR_BGR2GRAY };
use opencv::prelude::{ MatTrait, Vector };
use std::error::Error;
use std::vec::Vec;

#[path = "../base/opencv_utils.rs"] mod opencv_utils;
#[path = "../base/math_utils.rs"] mod math_utils;

use opencv_utils::{ get_pixel };

pub fn harris_detect_corner(src: &Mat,
                            block_size: i32,
                            k: f64,
                            threshold: f32,
                            cut_edge: bool) -> Result<Vec<Point>, Box<dyn Error>> {
    let mut buffer: Mat = Mat::default()?;
    let mut gray_image: Mat = Mat::default()?;
    opencv::imgproc::cvt_color(src, &mut gray_image, COLOR_BGR2GRAY, 0).unwrap();

    // Step1: Compute Ix, Iy
    let mut ix: Mat = Mat::default()?;
    let mut iy: Mat = Mat::default()?;
    let mut buffer_x = Mat::default()?;
    let mut buffer_y = Mat::default()?;

    spatial_gradient(&gray_image, &mut buffer_x, &mut buffer_y, 3, BORDER_DEFAULT).unwrap();
    buffer_x.convert_to(&mut ix, CV_32FC1, 1.0, 0.0).unwrap();
    buffer_y.convert_to(&mut iy, CV_32FC1, 1.0, 0.0).unwrap();
    
    // Step2: Compute Ix^2 Iy^2 IxIy
    let mut ix2: Mat = Mat::default()?;
    opencv::core::multiply(&ix, &ix, &mut ix2, 1.0, -1).unwrap();
    let mut iy2: Mat = Mat::default()?;
    opencv::core::multiply(&iy, &iy, &mut iy2, 1.0, -1).unwrap();
    let mut ixiy: Mat = Mat::default()?;
    opencv::core::multiply(&ix, &iy, &mut ixiy, 1.0, -1).unwrap();

    // Step3: Gaussian filter on ix2, iy2, ixiy
    let mut sx2: Mat = Mat::default()?;
    let mut sy2: Mat = Mat::default()?;
    let mut sxsy: Mat = Mat::default()?;
    gaussian_blur(&ix2, &mut sx2, Size::new(block_size, block_size), 
                  1.0, 0.0, BORDER_DEFAULT).unwrap();
    gaussian_blur(&iy2, &mut sy2, Size::new(block_size, block_size), 
                  1.0, 0.0, BORDER_DEFAULT).unwrap();
    gaussian_blur(&ixiy, &mut sxsy, Size::new(block_size, block_size), 
                  1.0, 0.0, BORDER_DEFAULT).unwrap();

    // Step4: Now that M = [sx2, sxsy]
    //                     [sxsy, sy2]
    // We compute the R = det(M) - k * (trace(M))^2
    let mut sx2sy2 = Mat::default()?;
    opencv::core::multiply(&sx2, &sy2, &mut sx2sy2, 1.0, -1).unwrap();
    let mut sxsy2 = Mat::default()?;
    opencv::core::multiply(&sxsy, &sxsy, &mut sxsy2, 1.0, -1).unwrap();
    let mut det = Mat::default()?;
    opencv::core::add_weighted(&sx2sy2, 1.0, &sxsy2, -1.0, 0.0, &mut det, -1).unwrap();
    opencv::core::add_weighted(&sx2, 1.0, &sy2, 1.0, 0.0, &mut buffer, -1).unwrap();
    let mut trace = Mat::default()?;
    opencv::core::multiply(&buffer, &buffer, &mut trace, 1.0, -1).unwrap();
    let mut R = Mat::default()?;
    opencv::core::add_weighted(&det, 1.0, &trace, -k, 0.0, &mut buffer, -1).unwrap();
    opencv::core::normalize(&buffer, &mut R, 0.0, 255.0, 
                            NORM_MINMAX, -1, &Mat::default()?).unwrap();

    // Now we do the output
    let rows = src.rows();
    let cols = src.cols();

    let mx = [-1, -1, -1, 0, 0, 1, 1, 1];
    let my = [-1, 0, 1, 1, -1, 0, 1, -1];
    let mut feature_num: i32 = 0;
    let mut out_feature: Vec<Point> = Vec::new();
    let mut thresh: i32 = 0;
    if cut_edge == true {
        thresh = 8;
    }
    for i in thresh..rows-thresh {
        for j in thresh..cols-thresh {
            let pixel = get_pixel::<f32>(&R, i, j);
            if pixel > threshold {
                let mut is_local_maximum = true;
                for k in 0..8 {
                    let rr = i + mx[k];
                    let cc = j + my[k];
                    if rr >=0 && rr < rows && cc >=0 && cc < cols {
                        if get_pixel::<f32>(&R, rr, cc) > pixel {
                            is_local_maximum = false;
                            break;
                        }
                    }
                }
                if is_local_maximum == true {
                    feature_num += 1;
                    out_feature.push(Point::new(j, i));
                }
            }
        }
    }

    log::trace!("Detected features: {}.", feature_num);

    Ok(out_feature)
} 
