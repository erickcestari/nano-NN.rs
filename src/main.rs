use clap::ArgGroup;
use image::{Luma, imageops};
use nano_nn::EPOCHS;
use nano_nn::IMAGE_HEIGHT;
use nano_nn::IMAGE_WIDTH;
use nano_nn::PIXEL_SCALE;
use nano_nn::gui::DigitApp;

use nano_nn::{
    INPUT_SIZE, TEST_IMAGES_PATH, TEST_LABELS_PATH, TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH,
    dataset::Dataset, neural_network::NeuralNetwork,
};

use clap::{Arg, Command};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("nano-nn")
        .version("1.0")
        .author("Erick Cestari")
        .about("MNIST digit recognition CLI with optional GUI mode")
        .arg(
            Arg::new("gui")
                .long("gui")
                .help("Run interactive GUI mode")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("image")
                .long("image")
                .help("Path to the input image for digit recognition")
                .value_name("IMAGE_PATH")
                .required(false),
        )
        .arg(
            Arg::new("train")
                .long("train")
                .help("Train the neural network model")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("model")
                .long("model")
                .help("Path to the neural network model file (JSON)")
                .value_name("MODEL_PATH")
                .required(false),
        )
        .group(
            ArgGroup::new("mode")
                .args(["gui", "image", "train"])
                .required(true)
                .multiple(false),
        )
        .arg_required_else_help(true)
        .get_matches();

    let gui = matches.get_flag("gui");

    if gui {
        let model_path = matches
            .get_one::<String>("model")
            .map(String::as_str)
            .unwrap_or("model.json");
        let nn = NeuralNetwork::import_model(model_path, leaky_relu, leaky_relu_derivative)?;
        return run_gui(nn);
    }

    let train = matches.get_flag("train");

    if train {
        train_and_save_model(leaky_relu, leaky_relu_derivative)?;
        return Ok(());
    }

    let image_path = matches
        .get_one::<String>("image")
        .expect("Image path is required");
    println!("Loading image: {}", image_path);
    let image_pixels = load_image(image_path)?;

    let mut nn = if let Some(model_path) = matches.get_one::<String>("model") {
        println!("Loading model: {}", model_path);
        NeuralNetwork::import_model(model_path, leaky_relu, leaky_relu_derivative)?
    } else {
        println!("Initializing neural network and training...");
        train_and_save_model(leaky_relu, leaky_relu_derivative)?
    };

    predict_and_display(&mut nn, &image_pixels);

    Ok(())
}

fn train_and_save_model<F, D>(
    activation: F,
    activation_derivative: D,
) -> Result<NeuralNetwork<F, D>, Box<dyn std::error::Error>>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    println!("\nLoading MNIST datasets...");
    let train_data = Dataset::load(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH)?;
    let val_data = Dataset::load(TEST_IMAGES_PATH, TEST_LABELS_PATH)?;

    train_data.print_info("Training");
    val_data.print_info("Validation");

    let mut nn = NeuralNetwork::new(activation, activation_derivative);

    println!("\nTraining model for {} epochs...", EPOCHS);
    nn.train(&train_data, &val_data, EPOCHS);

    println!("\nExporting model...");
    nn.export_model("model.json")?;

    Ok(nn)
}

fn predict_and_display<F: Fn(f64) -> f64, D: Fn(f64) -> f64>(
    nn: &mut NeuralNetwork<F, D>,
    image_pixels: &[f64],
) {
    println!("\nMaking prediction on input image...");
    let probs = nn.forward(image_pixels);

    println!("\nPrediction results:");
    println!("{}", "-".repeat(30));

    let mut top_predictions: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
    top_predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for &(digit, prob) in top_predictions.iter().take(3) {
        println!("Digit {}: {:.2}%", digit, prob * 100.0);
    }

    let (best_digit, best_prob) = top_predictions[0];
    println!("{}", "-".repeat(30));
    println!(
        "\nThe digit is probably {} with {:.2}% confidence.",
        best_digit,
        best_prob * 100.0
    );
}

fn run_gui<F: Fn(f64) -> f64, D: Fn(f64) -> f64>(
    nn: NeuralNetwork<F, D>,
) -> Result<(), Box<dyn std::error::Error>> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "nano-nn",
        options,
        Box::new(|_cc| Ok(Box::new(DigitApp::new(nn)))),
    )?;
    Ok(())
}

fn load_image(filename: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let img = image::open(filename)?;
    let gray_img = img.into_luma8();

    let resized = imageops::resize(
        &gray_img,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        imageops::FilterType::Lanczos3,
    );

    let mut normalized_pixels = Vec::with_capacity(INPUT_SIZE);
    for pixel in resized.pixels() {
        let Luma([value]) = *pixel;
        normalized_pixels.push(value as f64 * PIXEL_SCALE);
    }

    Ok(normalized_pixels)
}

fn leaky_relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.01 * x }
}

fn leaky_relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.01 }
}
