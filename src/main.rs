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
#[path = "./core/default_feature_matcher.rs"] mod default_feature_matcher;
#[path = "./core/default_image_blender.rs"] mod image_blender;
#[path = "./core/default_image_matcher.rs"] mod image_matcher;
#[path = "./core/harris_corner_detector.rs"] mod harris_corner_detector;
#[path = "./core/mtb_image_alignment.rs"] mod mtb;
#[path = "./core/photographic_global_tone_mapping.rs"] mod global_tone_mapping;
#[path = "./core/photographic_local_tone_mapping.rs"] mod local_tone_mapping;
#[path = "./core/sift_feature_descriptor.rs"] mod sift;

fn main() -> Result<(), Box<dyn Error>> {
    pretty_env_logger::init();
    log::trace!("HDR-Rust Starts.");

    let image1: Mat = imread("/Users/apple/Pictures/parrington/prtn02.jpg", 1)?;
    let image2: Mat = imread("/Users/apple/Pictures/parrington/prtn01.jpg", 1)?;
    let image3: Mat = imread("/Users/apple/Pictures/parrington/prtn00.jpg", 1)?;
    let mut dst1: Mat = Mat::default()?;
    let mut indicies1: Mat = Mat::default()?;
    cy_wrap::cylindrial_wrap(&image1, 705.849, &mut dst1, &mut indicies1).unwrap();
    let mut dst2 = Mat::default()?;
    let mut indicies2: Mat = Mat::default()?;
    cy_wrap::cylindrial_wrap(&image2, 706.286, &mut dst2, &mut indicies2).unwrap();
    let mut dst3 = Mat::default()?;
    let mut indicies3: Mat = Mat::default()?;
    cy_wrap::cylindrial_wrap(&image3, 704.916, &mut dst3, &mut indicies3).unwrap();
    let out_features1 = harris_corner_detector::harris_detect_corner(&dst1, 3, 0.04, 72.0, true).unwrap();
    let out_features2 = harris_corner_detector::harris_detect_corner(&dst2, 3, 0.04, 64.0, true).unwrap();
    let out_features3 = harris_corner_detector::harris_detect_corner(&dst3, 3, 0.04, 64.0, true).unwrap();

    let mut features1: Mat = Mat::default()?;
    sift::sift_feature_description(&dst1, &out_features1, &mut features1).unwrap();
    let mut features2: Mat = Mat::default()?;
    sift::sift_feature_description(&dst2, &out_features2, &mut features2).unwrap();
    let mut features3: Mat = Mat::default()?;
    sift::sift_feature_description(&dst3, &out_features3, &mut features3).unwrap();
    let feature_match1 = default_feature_matcher::match_feature(&features1, &features2, 0.6).unwrap();
    let feature_match2 = default_feature_matcher::match_feature(&features2, &features3, 0.7).unwrap();

    let m1 = image_matcher::match_image(&mut dst1, &mut dst2, &out_features1, &out_features2, &feature_match1).unwrap();
    let m2 = image_matcher::match_image(&mut dst2, &mut dst3, &out_features2, &out_features3, &feature_match2).unwrap();

    let mut images: Vec<Mat> = Vec::new();
    images.push(dst1);
    images.push(dst2);
    images.push(dst3);
    let mut wrapped_image_indicies: Vec<Mat> = Vec::new();
    wrapped_image_indicies.push(indicies1);
    wrapped_image_indicies.push(indicies2);
    wrapped_image_indicies.push(indicies3);
    let mut alignments: Vec<Point> = Vec::new();
    alignments.push(m1);
    alignments.push(m2);

    let mut panorama = Mat::default()?;
    image_blender::blend_image(&images, &wrapped_image_indicies, &alignments, &mut panorama).unwrap();

    log::trace!("Sift feature extraction finished.");

     imwrite("/Users/apple/Desktop/panorma.jpg", &panorama, &VectorOfi32::new()).unwrap();
    // imwrite("/home/yucwang/Desktop/cy.jpg", &dst, &VectorOfi32::new()).unwrap();
    imwrite("/Users/apple/Desktop/harris_out1.jpg", &images[0], &VectorOfi32::new()).unwrap();
    imwrite("/Users/apple/Desktop/harris_out2.jpg", &images[1], &VectorOfi32::new()).unwrap();
    imwrite("/Users/apple/Desktop/harris_out3.jpg", &images[2], &VectorOfi32::new()).unwrap();
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
