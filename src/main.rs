use opencv::core::Mat;
use opencv::imgcodecs::{imread, imwrite};
use opencv::imgproc::{cvt_color, COLOR_RGB2GRAY};
use opencv::prelude::Vector;
use opencv::types::VectorOfi32;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let img: Mat = imread("/home/yucwang/Pictures/test_pictures/haibara_1.jpg", 1)
        .expect("Input pictures failed.");

    let mut gray_img = Mat::default()?;
    cvt_color(&img, &mut gray_img, COLOR_RGB2GRAY, 0)?;

    let img_write_types = VectorOfi32::with_capacity(0);
    imwrite("/home/yucwang/Desktop/haibara_2_transformed.jpg", &img, &img_write_types)
        .expect("Output pictures failed.");

    Ok(())
}
