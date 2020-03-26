extern crate pretty_env_logger;
extern crate log;

use opencv::core::Mat;
use opencv::imgcodecs::{imread, imwrite};
use opencv::prelude::Vector;
use opencv::types::VectorOfi32;

use std::error::Error;

#[path = "./base/opencv_utils.rs"] mod opencv_utils;

fn main() -> Result<(), Box<dyn Error>> {
    pretty_env_logger::init();
    log::trace!("HDR-Rust Starts.");

    let img: Mat = imread("/home/yucwang/Pictures/test_pictures/hdr_test_cases/1440px-StLouisArchMultExpEV-4.72.jpeg", 1)
        .expect("Input pictures failed.");

    let mut gray_img = Mat::default()?;
    opencv_utils::compute_mtb_image(&img, &mut gray_img).expect("compute mtb image failed.");

    let img_write_types = VectorOfi32::with_capacity(0);
    imwrite("/home/yucwang/Desktop/lotus_1.jpeg", &gray_img, &img_write_types)
        .expect("Output pictures failed.");

    log::trace!("HDR-Rust ends.");
    Ok(())
}
