use std::fs::File;
use std::io::{self, BufReader, Read};

use crate::{IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_SIZE, MAGIC_NUMBER_IMAGES, MAGIC_NUMBER_LABELS};

#[derive(Debug, Clone)]
pub struct Dataset {
    pub pixels: Vec<u8>,
    pub labels: Vec<u8>,
    pub num_images: usize,
    pub rows: u32,
    pub columns: u32,
    pub magic_number_images: u32,
    pub magic_number_labels: u32,
}

#[derive(Debug)]
pub enum DatasetError {
    IoError(io::Error),
    InvalidMagicNumber(String),
    MismatchedCounts(String),
    InvalidData(String),
}

impl From<io::Error> for DatasetError {
    fn from(error: io::Error) -> Self {
        DatasetError::IoError(error)
    }
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::IoError(e) => write!(f, "IO error: {}", e),
            DatasetError::InvalidMagicNumber(msg) => write!(f, "Invalid magic number: {}", msg),
            DatasetError::MismatchedCounts(msg) => write!(f, "Mismatched counts: {}", msg),
            DatasetError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
        }
    }
}

impl std::error::Error for DatasetError {}

impl Dataset {
    pub fn new(pixels: Vec<u8>, labels: Vec<u8>) -> Self {
        let num_images = labels.len();
        Dataset {
            pixels,
            labels,
            num_images,
            rows: IMAGE_HEIGHT as u32,
            columns: IMAGE_WIDTH as u32,
            magic_number_images: MAGIC_NUMBER_IMAGES,
            magic_number_labels: MAGIC_NUMBER_LABELS,
        }
    }

    pub fn load(images_path: &str, labels_path: &str) -> Result<Self, DatasetError> {
        let (pixels, num_images, rows, columns, magic_images) = Self::load_images(images_path)?;
        let (labels, num_labels, magic_labels) = Self::load_labels(labels_path)?;

        if num_images != num_labels {
            return Err(DatasetError::MismatchedCounts(format!(
                "Number of images ({}) and labels ({}) do not match",
                num_images, num_labels
            )));
        }

        println!(
            "Loaded dataset with {} images and {} labels",
            num_images, num_labels
        );

        Ok(Dataset {
            pixels,
            labels,
            num_images: num_images as usize,
            rows,
            columns,
            magic_number_images: magic_images,
            magic_number_labels: magic_labels,
        })
    }

    fn load_images(filepath: &str) -> Result<(Vec<u8>, u32, u32, u32, u32), DatasetError> {
        let file = File::open(filepath)?;
        let mut reader = BufReader::new(file);

        // Read magic number
        let magic_number = Self::read_u32_be(&mut reader)?;
        if magic_number != MAGIC_NUMBER_IMAGES {
            return Err(DatasetError::InvalidMagicNumber(format!(
                "Invalid magic number for images file: {} (expected {})",
                magic_number, MAGIC_NUMBER_IMAGES
            )));
        }

        // Read metadata
        let num_images = Self::read_u32_be(&mut reader)?;
        let rows = Self::read_u32_be(&mut reader)?;
        let columns = Self::read_u32_be(&mut reader)?;

        // Read pixel data
        let pixel_count = (num_images * rows * columns) as usize;
        let mut pixels = vec![0u8; pixel_count];
        reader.read_exact(&mut pixels)?;

        Ok((pixels, num_images, rows, columns, magic_number))
    }

    // Load labels from MNIST label file
    fn load_labels(filepath: &str) -> Result<(Vec<u8>, u32, u32), DatasetError> {
        let file = File::open(filepath)?;
        let mut reader = BufReader::new(file);

        // Read magic number
        let magic_number = Self::read_u32_be(&mut reader)?;
        if magic_number != MAGIC_NUMBER_LABELS {
            return Err(DatasetError::InvalidMagicNumber(format!(
                "Invalid magic number for labels file: {} (expected {})",
                magic_number, MAGIC_NUMBER_LABELS
            )));
        }

        // Read number of labels
        let num_labels = Self::read_u32_be(&mut reader)?;

        // Read label data
        let mut labels = vec![0u8; num_labels as usize];
        reader.read_exact(&mut labels)?;

        Ok((labels, num_labels, magic_number))
    }

    // Read u32 in big-endian format
    fn read_u32_be(reader: &mut impl Read) -> Result<u32, io::Error> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(u32::from_be_bytes(buf))
    }

    // Create a batch from the dataset
    pub fn create_batch(&self, batch_size: usize, batch_idx: usize) -> Option<Dataset> {
        let start = batch_idx * batch_size;
        if start >= self.num_images {
            return None;
        }

        let end = (start + batch_size).min(self.num_images);
        let batch_images = end - start;

        let pixels_start = start * INPUT_SIZE;
        let pixels_end = end * INPUT_SIZE;

        Some(Dataset {
            pixels: self.pixels[pixels_start..pixels_end].to_vec(),
            labels: self.labels[start..end].to_vec(),
            num_images: batch_images,
            rows: self.rows,
            columns: self.columns,
            magic_number_images: self.magic_number_images,
            magic_number_labels: self.magic_number_labels,
        })
    }

    // Print dataset information
    pub fn print_info(&self, dataset_name: &str) {
        println!("{} DATASET", dataset_name.to_uppercase());
        println!("Magic number (images): {}", self.magic_number_images);
        println!("Number of images: {}", self.num_images);
        println!("Image dimensions: {} x {}\n", self.rows, self.columns);
        println!("Magic number (labels): {}", self.magic_number_labels);
        println!("Number of labels: {}", self.labels.len());
        println!("\n");
    }
}
