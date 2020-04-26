extern crate pretty_env_logger;
extern crate log;

use opencv::core::{Mat, Point, Scalar};
use opencv::imgcodecs::{imread, imwrite};
use opencv::prelude::Vector;
use opencv::types::{VectorOfMat, VectorOfi32};

use std::error::Error;

#[path = "./base/math_utils.rs"] mod math_utils;
#[path = "./base/opencv_utils.rs"] mod opencv_utils;
#[path = "./core/cylindrical_image_wrapper.rs"] mod cy_wrap;
#[path = "./core/debevec_crf_solver.rs"] mod debevec_crf;
#[path = "./core/harris_corner_detector.rs"] mod harris_corner_detector;
#[path = "./core/mtb_image_alignment.rs"] mod mtb;
#[path = "./core/photographic_global_tone_mapping.rs"] mod global_tone_mapping;
#[path = "./core/photographic_local_tone_mapping.rs"] mod local_tone_mapping;

fn main() -> Result<(), Box<dyn Error>> {
    pretty_env_logger::init();
    log::trace!("HDR-Rust Starts.");

    let image: Mat = imread("/home/yucwang/Pictures/parrington/prtn01.jpg", 1)?;
    let mut dst: Mat = Mat::default()?;
    cy_wrap::cylindrial_wrap(&image, 706.2, &mut dst).unwrap();
    let out_features = harris_corner_detector::harris_detect_corner(&dst, 3, 0.04, 64.0).unwrap();

    for i in 0..out_features.len() {
        opencv::imgproc::circle(&mut dst, Point::new(out_features[i].to_vec2()[1], out_features[i].to_vec2()[0]), 
                                5, Scalar::new(0.0, 255.0, 0.0, 1.0), 1, 8, 0).unwrap();
    }

    imwrite("/home/yucwang/Desktop/cy.jpg", &dst, &VectorOfi32::new()).unwrap();
    imwrite("/home/yucwang/Desktop/harris_out.jpg", &dst, &VectorOfi32::new()).unwrap();
    //harris_corner_detector::harris_detect_corner(&image, 3, 0.05, 108.0).unwrap();

//    let mut images: VectorOfMat = VectorOfMat::new();
//    let image1: Mat = imread("/home/yucwang/I_love_hefei_50/IMG_20200402_183757.jpg", 1)?;
//    let image2: Mat = imread("/home/yucwang/I_love_hefei_50/IMG_20200402_183928.jpg", 1)?;
//    let image3: Mat = imread("/home/yucwang/I_love_hefei_50/IMG_20200402_184101.jpg", 1)?;
//    // let image4: Mat = imread("/home/yucwang/Pictures/i_love_hefei/i_love_hefei_3.jpg", 1)?;
//    images.push(image1);
//    images.push(image2);
//    images.push(image3);
//    // images.push(image4);
//
//    log::trace!("Starting align images.");
//    let mut out_aligned_images = VectorOfMat::new();
//    mtb::align(&images, &mut out_aligned_images, 8).unwrap();
//
//    images.clear();
//
//    log::trace!("Starting solving CRF.");
//    let mut out_hdri: Mat = Mat::default()?;
//    let mut shutter_speeds: Vec<f32> = Vec::new();
//    shutter_speeds.push(0.0182);
//    shutter_speeds.push(0.0667);
//    shutter_speeds.push(1.0);
//    // shutter_speeds.push(2.0);
//
//    debevec_crf::solve(&out_aligned_images, &shutter_speeds, 512, 0.7, &mut out_hdri).unwrap();
//
//    log::trace!("Starting output images.");
//    opencv_utils::save_exr_with_default(&std::string::String::from("/home/yucwang/Desktop/i_love_hefei_hdr.exr"), &out_hdri)?;
//
//    log::trace!("HDR-Rust ends.");
    Ok(())
}
