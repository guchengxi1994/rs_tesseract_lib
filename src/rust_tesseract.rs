use image::RgbImage;
use multimap::MultiMap;
use ndarray::Array3;
use polars::prelude::*;
use std::collections::HashMap;
use std::env::current_dir;
use std::fmt;
use std::fs;
use std::io::BufRead;
use std::os::windows::process::CommandExt;
use std::process::{Command, Stdio};
use std::string::ToString;

use crate::error::ImageFormatError;
use crate::error::ImageNotFoundError;
use crate::error::VersionError;
use crate::TESSERACT;

const FORMATS: [&'static str; 10] = [
    "JPEG", "JPG", "PNG", "PBM", "PGM", "PPM", "TIFF", "BMP", "GIF", "WEBP",
];

pub struct TesseractPath {
    pub path: Option<String>,
}

impl TesseractPath {
    pub fn new() -> TesseractPath {
        return TesseractPath { path: None };
    }

    pub fn use_current_dir() -> TesseractPath {
        let p = current_dir();

        match p {
            Ok(_p) => {
                if cfg!(target_os = "windows") {
                    let r = format!(
                        "{}/tesseract/tesseract.exe",
                        _p.as_os_str()
                            .to_str()
                            .unwrap_or("./tesseract/tesseract.exe")
                    );
                    return TesseractPath {
                        path: Some(String::from(r)),
                    };
                } else {
                    let r = format!(
                        "{}/tesseract/tesseract",
                        _p.as_os_str().to_str().unwrap_or("./tesseract/tesseract")
                    );
                    return TesseractPath {
                        path: Some(String::from(r)),
                    };
                }
            }
            Err(_) => {
                return TesseractPath::new();
            }
        }
    }

    pub fn use_certain_path(s: String) -> TesseractPath {
        return TesseractPath { path: Some(s) };
    }

    pub fn use_default() -> TesseractPath {
        if cfg!(target_os = "windows") {
            return TesseractPath {
                path: Some(String::from("tesseract.exe")),
            };
        } else {
            return TesseractPath {
                path: Some(String::from("tesseract")),
            };
        }
    }

    pub fn set_tesseract_path(&mut self, s: String) {
        self.path = Some(s);
    }
}

pub fn get_tesseract_installed_path() -> Option<String> {
    match TESSERACT.read() {
        Ok(t) => {
            return t.path.clone();
        }
        Err(_) => None,
    }
}

pub fn set_tesseract_installed_path(s: String) {
    let t = TESSERACT.write();
    match t {
        Ok(mut t0) => t0.set_tesseract_path(s),
        Err(_) => {
            println!("error reset path")
        }
    }
}

fn check_if_installed() -> bool {
    let p = get_tesseract_installed_path();
    match p {
        Some(p0) => {
            if cfg!(target_os = "windows") {
                match Command::new(p0).stdout(Stdio::null()).spawn() {
                    Ok(_) => return true,
                    Err(_) => return false,
                }
            } else {
                match Command::new(p0).stdout(Stdio::null()).spawn() {
                    Ok(_) => return true,
                    Err(_e) => return false,
                }
            }
        }
        None => false,
    }
}

pub struct ModelOutput {
    pub output_info: String,
    pub output_bytes: Vec<u8>,
    pub output_dict: MultiMap<String, String>,
    pub output_string: String,
    pub output_dataframe: Vec<Series>,
}

impl ModelOutput {
    fn new() -> ModelOutput {
        ModelOutput {
            output_info: String::new(),
            output_bytes: Vec::new(),
            output_dict: MultiMap::new(),
            output_string: String::new(),
            output_dataframe: Vec::new(),
        }
    }
}

impl fmt::Display for ModelOutput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.output_string)
    }
}

#[derive(Clone)]
pub struct Args {
    pub out_filename: &'static str,
    pub lang: &'static str,
    pub config: HashMap<&'static str, &'static str>,
    pub dpi: i32,
    pub boxfile: bool,
}

impl Args {
    pub fn new() -> Args {
        Args {
            config: HashMap::new(),
            lang: "eng",
            out_filename: "out",
            dpi: 150,
            boxfile: false,
        }
    }
}

#[derive(Clone)]
pub struct Image {
    pub path: String,
    pub ndarray: Array3<u8>,
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path)
    }
}

impl Image {
    pub fn new(path: String, ndarray: Array3<u8>) -> Image {
        Image { path, ndarray }
    }

    fn is_empty_ndarray(&self) -> bool {
        let mut is_empty: bool = true;
        for _elem in &self.ndarray {
            is_empty = false;
        }
        return is_empty;
    }

    fn size_of_ndarray(&self) -> (usize, usize, usize) {
        return self.ndarray.dim();
    }

    fn ndarray_to_image(self) -> RgbImage {
        let (height, width, _) = self.size_of_ndarray();
        let raw = self.ndarray.into_raw_vec();

        RgbImage::from_raw(width as u32, height as u32, raw)
            .expect("Couldnt convert ndarray to RgbImage.")
    }
}

fn type_of<T>(_: &T) -> String {
    let t = String::from(std::any::type_name::<T>());
    return t;
}

fn read_output_file(filename: &String) -> String {
    let f = fs::read_to_string(filename.to_owned())
        .expect("File reading error. Filename does not exist.");

    return f;
}

fn check_image_format(img: &Image) -> bool {
    let splits: Vec<&str> = img.path.split(".").collect();
    let format = splits.last().unwrap().to_string();
    let tmp = String::from(&format).to_uppercase();
    let tmp2 = String::from(&format).to_lowercase();
    let uppercase_format = tmp.as_str();
    let format = tmp2.as_str();

    if FORMATS.contains(&format) || FORMATS.contains(&uppercase_format) {
        return true;
    } else {
        return false;
    }
}

pub fn get_tesseract_version() -> String {
    let is_installed: bool = check_if_installed();
    if !is_installed {
        return String::new();
    }

    let p = get_tesseract_installed_path();
    match p {
        Some(p0) => {
            let command = Command::new(p0)
                .creation_flags(0x08000000)
                .arg("--version")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            let output = command.wait_with_output().unwrap();

            let out = output.stdout;
            let err = output.stderr;
            let status = output.status;

            match status.code() {
                Some(code) => println!("Exited with status code: {}", code),
                None => println!("Exited with error: {}", VersionError),
            }

            let mut str_res = String::new();
            if out.len() == 0 {
                err.lines()
                    .for_each(|line| str_res = format!("{}\n{}", str_res, line.unwrap()));
            } else {
                out.lines()
                    .for_each(|line| str_res = format!("{}\n{}", str_res, line.unwrap()));
            }

            return str_res;
        }
        None => String::new(),
    }
}

pub fn image_to_data(image: &Image, args: Args) -> ModelOutput {
    let str_out: ModelOutput = image_to_string(&image, args.clone());

    let mut box_args = args.clone();
    box_args.boxfile = true;
    let box_out: ModelOutput = image_to_boxes(&image, box_args);

    let out = ModelOutput {
        output_info: str_out.output_info,
        output_bytes: str_out.output_bytes,
        output_dict: box_out.output_dict,
        output_string: str_out.output_string,
        output_dataframe: box_out.output_dataframe,
    };

    let mut tesstable_args = args.clone();
    tesstable_args.config.insert("-c", "tessedit_create_tsv=1");
    let _tesstable = run_tesseract(&image, &tesstable_args);

    if check_image_format(&image) {
        return out;
    } else {
        println!("{:?}", "image error");
        return ModelOutput::new();
    }
}

pub fn image_to_boxes(image: &Image, args: Args) -> ModelOutput {
    let r = run_tesseract(&image, &args);
    match r {
        Some(r0) => r0,
        None => ModelOutput::new(),
    }
}

pub fn image_to_string(image: &Image, args: Args) -> ModelOutput {
    let r = run_tesseract(&image, &args);
    match r {
        Some(r0) => r0,
        None => ModelOutput::new(),
    }
}

fn run_tesseract(image: &Image, args: &Args) -> Option<ModelOutput> {
    // check if tesseract is installed
    let is_installed: bool = check_if_installed();
    if !is_installed {
        return None;
    }

    assert_eq!(type_of(&image.path), type_of(&String::new()));
    assert_eq!(
        type_of(&image.ndarray),
        type_of(&Array3::<u8>::zeros((0, 0, 0)))
    );

    // check if image path or ndarray is provided
    let mut image_arg = String::from("");
    let is_empty_ndarray = &image.is_empty_ndarray();
    if image.path.len() == 0 && !*is_empty_ndarray {
        // convert ndarray to rgbimage and save image in parent directory
        let tmp_img = image.clone();
        let i = tmp_img.ndarray_to_image();
        let working_dir = current_dir().unwrap().as_path().display().to_string();
        let new_path = [working_dir, String::from("ndarray_converted.png")].join("/");

        match i.save(&new_path) {
            Ok(_r) => {
                println!("Image saved: {:?}", new_path);
                image_arg = new_path;
            }
            Err(e) => println!("Error while saving image: {:?}", e),
        }
    }
    // both image path and ndarray are empty
    else if image.path.len() == 0 && *is_empty_ndarray {
        println!("{:?}", ImageNotFoundError);
        return None;
    }
    // path is filled
    else {
        if !check_image_format(&image) {
            println!("{:?}", ImageFormatError);
            return None;
        }
        image_arg = image.to_string().replace('"', "").to_owned();
    }

    for (key, value) in &args.config {
        println!("Configuration: {:?}:{:?}", key, value)
    }

    // check if boxmode is activated
    let mut boxarg = String::new();
    if args.boxfile {
        boxarg = String::from("makebox");
    }

    // check if tesstable command is given
    let mut tesstable_arg = "tessedit_create_tsv=0";
    if args.config.contains_key("-c") {
        tesstable_arg = args.config["-c"];
    }

    // check if psm and oem flags are set
    let mut psm = "3";
    let mut oem = "3";
    if args.config.contains_key("psm") {
        psm = args.config["psm"];
    }

    if args.config.contains_key("oem") {
        oem = args.config["oem"];
    }

    println!("the image arg is: {:?}", image_arg);

    let tess_path = get_tesseract_installed_path().unwrap();

    let command = if cfg!(target_os = "windows") {
        Command::new(tess_path)
            .creation_flags(0x08000000)
            .arg(image_arg)
            .arg(args.out_filename)
            .arg("-l")
            .arg(args.lang)
            .arg("--dpi")
            .arg(args.dpi.to_string())
            .arg("--psm")
            .arg(psm)
            .arg("--oem")
            .arg(oem)
            .arg("-c")
            .arg(tesstable_arg)
            .arg(boxarg)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap()
    } else {
        Command::new(tess_path)
            .arg(image_arg)
            .arg(args.out_filename)
            .arg("-l")
            .arg(args.lang)
            .arg("--dpi")
            .arg(args.dpi.to_string())
            .arg("--psm")
            .arg(psm)
            .arg("--oem")
            .arg(oem)
            .arg("-c")
            .arg(tesstable_arg)
            .arg(boxarg)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap()
    };

    let output = command.wait_with_output().unwrap();
    println!("{:?}", output);

    let out = output.stdout;
    let err = output.stderr;
    let status = output.status;

    match status.code() {
        Some(code) => println!("Exited with status code: {}", code),
        None => println!("Process terminated by signal"),
    }

    let mut str_res = String::new();
    if out.len() == 0 {
        err.lines()
            .for_each(|line| str_res = format!("{}\n{}", str_res, line.unwrap()));
    } else {
        out.lines()
            .for_each(|line| str_res = format!("{}\n{}", str_res, line.unwrap()));
    }

    // read tesseract output from output file "out.txt"
    let mut _out_f = String::new();
    if !args.boxfile {
        if !args.out_filename.contains(".txt") {
            _out_f = format!("{}.txt", args.out_filename);
        } else {
            _out_f = args.out_filename.to_string();
        }
    }
    // if boxfile is requested -> read from .box file
    else {
        if !args.out_filename.contains(".box") {
            _out_f = format!("{}.box", args.out_filename);
        } else {
            _out_f = args.out_filename.to_string();
        }
    }

    let file_output = read_output_file(&_out_f);

    // multimap used for box files -> stores character as key and box boundaries as value (or list of values)
    let mut dict = MultiMap::new();
    let mut df = Vec::new();
    if args.boxfile {
        for line in file_output.lines() {
            if line.contains(" ") {
                // fill dict
                let tuple = line.split_once(" ").unwrap();
                dict.insert(String::from(tuple.0), String::from(tuple.1));

                // fill DataFrame (Vec of Series)
                let character: &str = &tuple.0;
                let mut box_boundaries = Vec::new();
                for num in tuple.1.split(" ") {
                    let num_int: i32 = num.parse::<i32>().unwrap();
                    box_boundaries.push(num_int);
                }
                let tmp_series = Series::new(character, &box_boundaries);
                df.push(tmp_series);
            }
        }
    }

    let out = ModelOutput {
        output_info: str_res,
        output_bytes: file_output.as_bytes().to_vec(),
        output_dict: dict,
        output_string: file_output,
        output_dataframe: df,
    };

    return Some(out);
}

mod tests {

    #[test]
    fn current_dir_test() {
        let p = std::env::current_dir();
        match p {
            Ok(_p) => {
                println!("{:?}", _p.as_os_str().to_str())
            }
            Err(_) => {
                println!("error")
            }
        }
    }

    #[test]
    fn get_tesseract_version_test() {
        let p = String::from(r"D:\tesseract\tesseract.exe");
        super::set_tesseract_installed_path(p);
        let r = super::get_tesseract_version();
        println!("{:?}", r)
    }

    #[test]
    fn ocr_test() {
        let ip = String::from(r"D:\tesseract\tesseract.exe");
        let p = String::from(r"C:\Users\xiaoshuyui\Desktop\screenshot-20230209-130916.png");
        super::set_tesseract_installed_path(ip);

        let img = super::Image {
            path: String::from(p),
            ndarray: ndarray::Array3::<u8>::zeros((200, 200, 3)), // example: creates an 100x100 pixel image with 3 colour channels (RGB)
        };

        // default_args.lang = "chi_sim";
        let mut image_to_string_args = super::Args {
            out_filename: "out",
            lang: "chi_sim",
            config: std::collections::HashMap::new(),
            dpi: 150,
            boxfile: false,
        };

        image_to_string_args.config.insert("psm", "6");
        image_to_string_args.config.insert("oem", "3");

        let output = crate::rust_tesseract::image_to_string(&img, image_to_string_args);
        println!("\nThe String output is: {:?}", output.output_string);
    }
}
