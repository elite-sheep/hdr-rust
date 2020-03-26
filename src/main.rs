extern crate pretty_env_logger;
extern crate log;

use opencv::core::Mat;
use opencv::imgcodecs::{imread, imwrite};
use opencv::prelude::Vector;
use opencv::types::VectorOfi32;
use opencv::prelude::MatTrait;

use std::error::Error;

#[path = "./base/opencv_utils.rs"] mod opencv_utils;

fn main() -> Result<(), Box<dyn Error>> {
    pretty_env_logger::init();
    log::trace!("HDR-Rust Starts.");

    let img: Mat = imread("/home/yucwang/Pictures/test_pictures/haibara_1.jpg", 1)
        .expect("Input pictures failed.");

    // let mut gray_img = Mat::default()?;
    let gray_img = opencv_utils::compute_mtb_image(&img).expect("compute mtb image failed.");


    let pixel: i32 = gray_img.cols();
    let origin: i32 = img.cols();
    println!("{} {}.", origin, pixel);

    log::trace!("HDR-Rust Starts.");
    let img_write_types = VectorOfi32::with_capacity(0);
    imwrite("/home/yucwang/Desktop/haibara_2_transformed.bmp", &gray_img, &img_write_types)
        .expect("Output pictures failed.");

    log::trace!("HDR-Rust ends.");
    Ok(())
}
