// modified from https://github.com/thomasgruebl/rusty-tesseract

use std::sync::RwLock;

use rust_tesseract::{set_tesseract_installed_path, Image};

pub mod error;
pub mod rust_tesseract;

#[macro_use]
extern crate lazy_static;
lazy_static! {
    static ref TESSERACT: RwLock<rust_tesseract::TesseractPath> =
        RwLock::new(rust_tesseract::TesseractPath::new());
}

pub fn init_ocr_engine(p: String) {
    set_tesseract_installed_path(p);
}

pub fn get_string_from_image(img_path: String) -> String {
    let img = Image {
        path: String::from(img_path),
        ndarray: ndarray::Array3::<u8>::zeros((200, 200, 3)), // example: creates an 100x100 pixel image with 3 colour channels (RGB)
    };
    let mut image_to_string_args = rust_tesseract::Args {
        out_filename: "out",
        lang: "chi_sim",
        config: std::collections::HashMap::new(),
        dpi: 150,
        boxfile: false,
    };

    image_to_string_args.config.insert("psm", "6");
    image_to_string_args.config.insert("oem", "3");

    let output = crate::rust_tesseract::image_to_string(&img, image_to_string_args);
    return output.output_string;
}
