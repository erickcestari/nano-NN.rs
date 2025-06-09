pub mod dataset;
pub mod gui;
pub mod neural_network;

pub const INPUT_SIZE: usize = 784; // 28x28 pixels
const HIDDEN_SIZE_1: usize = 256; // number of neurons in the first hidden layer
const HIDDEN_SIZE_2: usize = 128; // number of neurons in the second hidden layer
const HIDDEN_SIZE_3: usize = 64; // number of neurons in the third hidden layer
const HIDDEN_SIZE_4: usize = 32; // number of neurons in the fourth hidden layer
pub const OUTPUT_SIZE: usize = 10; // number of classes (0-9)

// You can add more number of layers, if you want.
pub const LAYERS: &[usize] = &[
    INPUT_SIZE,
    HIDDEN_SIZE_1,
    HIDDEN_SIZE_2,
    HIDDEN_SIZE_3,
    HIDDEN_SIZE_4,
    OUTPUT_SIZE,
]; // number of neurons in each layer

pub const BATCH_SIZE: usize = 128;
pub const EPOCHS: usize = 20;
pub const PATIENCE_LIMIT: usize = 10;

pub const IMAGE_WIDTH: u32 = 28;
pub const IMAGE_HEIGHT: u32 = 28;
pub const PIXEL_SCALE: f64 = 1.0 / 255.0;
pub const MAGIC_NUMBER_IMAGES: u32 = 2051;
pub const MAGIC_NUMBER_LABELS: u32 = 2049;

pub const TRAIN_IMAGES_PATH: &str = "dataset/train-images.idx3-ubyte";
pub const TRAIN_LABELS_PATH: &str = "dataset/train-labels.idx1-ubyte";
pub const TEST_IMAGES_PATH: &str = "dataset/t10k-images.idx3-ubyte";
pub const TEST_LABELS_PATH: &str = "dataset/t10k-labels.idx1-ubyte";
