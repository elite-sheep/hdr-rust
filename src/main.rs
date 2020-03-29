extern crate pretty_env_logger;
extern crate log;

use opencv::core::Mat;
use opencv::imgcodecs::{imread, imwrite};
use opencv::prelude::Vector;
use opencv::types::{VectorOfi32, VectorOfMat};

use std::error::Error;

#[path = "./base/math_utils.rs"] mod math_utils;
#[path = "./core/debevec_crf_solver.rs"] mod debevec_crf;
#[path = "./core/mtb_image_alignment.rs"] mod mtb;

fn main() -> Result<(), Box<dyn Error>> {
    pretty_env_logger::init();
    log::trace!("HDR-Rust Starts.");

    let mut images: VectorOfMat = VectorOfMat::new();
    let image1: Mat = imread("/home/yucwang/Pictures/test_pictures/hdr_test_cases/1440px-StLouisArchMultExpEV-4.72.jpeg", 1)?;
    let image2: Mat = imread("/home/yucwang/Pictures/test_pictures/hdr_test_cases/1440px-StLouisArchMultExpEV-1.82.jpeg", 1)?;
    let image3: Mat = imread("/home/yucwang/Pictures/test_pictures/hdr_test_cases/1440px-StLouisArchMultExpEV+1.51.jpeg", 1)?;
    let image4: Mat = imread("/home/yucwang/Pictures/test_pictures/hdr_test_cases/1440px-StLouisArchMultExpEV+4.09.jpeg", 1)?;
    images.push(image1);
    images.push(image2);
    images.push(image3);
    images.push(image4);

    log::trace!("Starting align images.");
    let mut out_aligned_images = VectorOfMat::new();
    mtb::align(&images, &mut out_aligned_images, 6)?;

    log::trace!("Starting solving CRF.");
    let mut out_hdri: Mat = Mat::default()?;
    let mut shutter_speeds: Vec<f32> = Vec::new();
    shutter_speeds.push(0.03333);
    shutter_speeds.push(0.25);
    shutter_speeds.push(2.5);
    shutter_speeds.push(15.0);

    debevec_crf::solve(&out_aligned_images, &shutter_speeds, 2.0, &mut out_hdri);

    log::trace!("Starting output images.");
    let options: VectorOfi32 = VectorOfi32::new();
    imwrite("/home/yucwang/Desktop/lotus_hdr.exr", &out_hdri, &options);

    log::trace!("HDR-Rust ends.");
    Ok(())
}
