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
    let image1: Mat = imread("/home/yucwang/I_love_hefei_50/IMG_20200402_183757.jpg", 1)?;
    let image2: Mat = imread("/home/yucwang/I_love_hefei_50/IMG_20200402_183928.jpg", 1)?;
    let image3: Mat = imread("/home/yucwang/I_love_hefei_50/IMG_20200402_184101.jpg", 1)?;
    // let image4: Mat = imread("/home/yucwang/Pictures/i_love_hefei/i_love_hefei_3.jpg", 1)?;
    images.push(image1);
    images.push(image2);
    images.push(image3);
    // images.push(image4);

    log::trace!("Starting align images.");
    let mut out_aligned_images = VectorOfMat::new();
    mtb::align(&images, &mut out_aligned_images, 8).unwrap();

    images.clear();

    log::trace!("Starting solving CRF.");
    let mut out_hdri: Mat = Mat::default()?;
    let mut shutter_speeds: Vec<f32> = Vec::new();
    shutter_speeds.push(0.0182);
    shutter_speeds.push(0.0667);
    shutter_speeds.push(1.0);
    // shutter_speeds.push(2.0);

    debevec_crf::solve(&out_aligned_images, &shutter_speeds, 512, 0.7, &mut out_hdri).unwrap();

    log::trace!("Starting output images.");
    let mut options: VectorOfi32 = VectorOfi32::new();
    options.push(opencv::imgcodecs::IMWRITE_EXR_TYPE_FLOAT);
    imwrite("/home/yucwang/Desktop/i_love_hefei_hdr.exr", &out_hdri, &options).unwrap();

    log::trace!("HDR-Rust ends.");
    Ok(())
}
