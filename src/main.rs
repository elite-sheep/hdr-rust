extern crate pretty_env_logger;
extern crate log;

use opencv::core::Mat;
use opencv::imgcodecs::{imread, imwrite};
use opencv::prelude::Vector;
use opencv::types::{VectorOfi32, VectorOfMat};

use std::error::Error;

#[path = "./core/mtb_image_alignment.rs"] mod mtb;

fn main() -> Result<(), Box<dyn Error>> {
    pretty_env_logger::init();
    log::trace!("HDR-Rust Starts.");

    let mut images: VectorOfMat = VectorOfMat::new();
    let image1: Mat = imread("/home/yucwang/Pictures/test_pictures/hdr_test_cases/1440px-StLouisArchMultExpEV+1.51.jpeg", 1)?;
    let image2: Mat = imread("/home/yucwang/Pictures/test_pictures/hdr_test_cases/1440px-StLouisArchMultExpEV-1.82.jpeg", 1)?;
    let image3: Mat = imread("/home/yucwang/Pictures/test_pictures/hdr_test_cases/1440px-StLouisArchMultExpEV-4.72.jpeg", 1)?;
    let image4: Mat = imread("/home/yucwang/Pictures/test_pictures/hdr_test_cases/1440px-StLouisArchMultExpEV+1.18.jpeg", 1)?;
    let image5: Mat = imread("/home/yucwang/Pictures/test_pictures/hdr_test_cases/1440px-StLouisArchMultExpEV+4.09.jpeg", 1)?;
    images.push(image1);
    images.push(image2);
    images.push(image3);
    images.push(image4);
    images.push(image5);

    let mut out_aligned_images = VectorOfMat::new();
    mtb::align(&images, &mut out_aligned_images, 6)?;

    log::trace!("Starting output images.");
    let img_write_types = VectorOfi32::with_capacity(0);
    imwrite("/home/yucwang/Desktop/lotus_1.jpeg", &out_aligned_images.get(0)?, &img_write_types)?;
    imwrite("/home/yucwang/Desktop/lotus_2.jpeg", &out_aligned_images.get(1)?, &img_write_types)?;
    imwrite("/home/yucwang/Desktop/lotus_3.jpeg", &out_aligned_images.get(2)?, &img_write_types)?;
    imwrite("/home/yucwang/Desktop/lotus_4.jpeg", &out_aligned_images.get(3)?, &img_write_types)?;
    imwrite("/home/yucwang/Desktop/lotus_5.jpeg", &out_aligned_images.get(4)?, &img_write_types)?;

    log::trace!("HDR-Rust ends.");
    Ok(())
}
