use core::fmt;
use rand::{Rng, prelude::ThreadRng};
use serde::{Deserialize, Serialize};
use std::fs::File;

use crate::{
    BATCH_SIZE, INPUT_SIZE, LAYERS, OUTPUT_SIZE, PATIENCE_LIMIT, PIXEL_SCALE, dataset::Dataset,
};

pub type Matrix = Vec<Vec<f64>>;

#[derive(Serialize, Deserialize)]
struct ModelData {
    layers: Vec<Layer>,
}

/// Single layer in the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    weights: Matrix,
    biases: Vec<f64>,
}

impl Layer {
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut ThreadRng) -> Self {
        let factor = (2.0f64 / in_dim as f64).sqrt();
        let weights = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| (rng.random::<f64>() - 0.5) * factor)
                    .collect()
            })
            .collect();
        let biases = vec![0.0; out_dim];
        Self { weights, biases }
    }

    /// Matrix-vector product plus bias
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(row, &b)| row.iter().zip(input).fold(b, |sum, (&w, &i)| sum + w * i))
            .collect()
    }
}

pub struct ForwardCache {
    pub layer_outputs: Vec<Vec<f64>>,
}

pub struct NeuralNetwork<F, D>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    pub layers: Vec<Layer>,
    activation: F,
    activation_deriv: D,
}

impl<F, D> NeuralNetwork<F, D>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    /// Create a new neural network with specified layer dimensions
    pub fn new_with_dims(layer_dims: &[usize], activation: F, activation_deriv: D) -> Self {
        let mut rng = rand::rng();
        let layers = layer_dims
            .windows(2)
            .map(|window| Layer::new(window[0], window[1], &mut rng))
            .collect();

        Self {
            layers,
            activation,
            activation_deriv,
        }
    }

    /// Create with default architecture (for backward compatibility)
    pub fn new(activation: F, activation_deriv: D) -> Self {
        Self::new_with_dims(LAYERS, activation, activation_deriv)
    }

    pub fn export_model(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let model_data = ModelData {
            layers: self.layers.clone(),
        };
        let mut file = File::create(filename)?;
        serde_json::to_writer(&mut file, &model_data)?;

        Ok(())
    }

    pub fn import_model(
        filename: &str,
        activation: F,
        activation_deriv: D,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let model_data: ModelData = serde_json::from_reader(&file)?;

        Ok(Self {
            layers: model_data.layers,
            activation,
            activation_deriv,
        })
    }

    pub fn forward_cached(&mut self, pixels: &[f64]) -> ForwardCache {
        let mut layer_outputs = Vec::with_capacity(self.layers.len() + 1);
        layer_outputs.push(pixels.to_vec());

        let mut current_input = pixels.to_vec();

        for (i, layer) in self.layers.iter().enumerate() {
            let mut output = layer.forward(&current_input);

            if i < self.layers.len() - 1 {
                output.iter_mut().for_each(|x| *x = (self.activation)(*x));
            } else {
                // Apply softmax to the last layer
                output = softmax(&output);
            }

            layer_outputs.push(output.clone());
            current_input = output;
        }

        ForwardCache { layer_outputs }
    }

    pub fn forward(&mut self, pixels: &[f64]) -> Vec<f64> {
        let cache = self.forward_cached(pixels);
        cache.layer_outputs.last().unwrap().clone()
    }

    pub fn train(&mut self, train: &Dataset, val: &Dataset, epochs: usize) {
        let mut best_val_loss = f64::INFINITY;
        let mut patience = 0;
        let lr = 0.001;
        let batches = (train.num_images + BATCH_SIZE - 1) / BATCH_SIZE;

        for epoch in 0..epochs {
            for batch_idx in 0..batches {
                if let Some(batch) = train.create_batch(BATCH_SIZE, batch_idx) {
                    batch
                        .pixels
                        .chunks(INPUT_SIZE)
                        .zip(&batch.labels)
                        .for_each(|(pix, &lbl)| {
                            let norm: Vec<f64> =
                                pix.iter().map(|&b| b as f64 * PIXEL_SCALE).collect();
                            let cache = self.forward_cached(&norm);

                            // Backpropagation
                            let output = cache.layer_outputs.last().unwrap();

                            let d_out: Vec<f64> = output
                                .iter()
                                .zip(
                                    (0..OUTPUT_SIZE)
                                        .map(|i| if i == lbl as usize { 1.0 } else { 0.0 }),
                                )
                                .map(|(&o, y)| o - y)
                                .collect();

                            let mut deltas = vec![d_out];

                            for i in (0..self.layers.len() - 1).rev() {
                                let delta = backprop_layer(
                                    &deltas[0],
                                    &self.layers[i + 1].weights,
                                    &cache.layer_outputs[i + 1],
                                    &self.activation_deriv,
                                );
                                deltas.insert(0, delta);
                            }

                            for (i, layer) in self.layers.iter_mut().enumerate() {
                                update_params(layer, &cache.layer_outputs[i], &deltas[i], lr);
                            }
                        });
                }
            }

            let (val_loss, accuracy) = self.validate(val);
            println!(
                "Epoch {} - ValLoss: {:.4}, ValAcc: {:.2}%",
                epoch,
                val_loss,
                accuracy * 100.0
            );

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                patience = 0;
            } else if patience >= PATIENCE_LIMIT {
                println!("Early stopping at epoch {}", epoch);
                break;
            } else {
                patience += 1;
            }
        }
    }

    fn validate(&mut self, val: &Dataset) -> (f64, f64) {
        let mut loss = 0.0;
        let mut correct = 0;

        val.pixels
            .chunks(INPUT_SIZE)
            .zip(&val.labels)
            .for_each(|(pix, &lbl)| {
                let norm: Vec<f64> = pix.iter().map(|&b| b as f64 * PIXEL_SCALE).collect();
                let preds = self.forward(&norm);
                let y_true: Vec<f64> = (0..OUTPUT_SIZE)
                    .map(|i| if i == lbl as usize { 1.0 } else { 0.0 })
                    .collect();
                loss += cross_entropy(&y_true, &preds);
                if argmax(&preds) == lbl as usize {
                    correct += 1;
                }
            });

        (
            loss / val.num_images as f64,
            correct as f64 / val.num_images as f64,
        )
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get the architecture (layer dimensions) of the network
    pub fn architecture(&self) -> Vec<usize> {
        let mut dims = Vec::with_capacity(self.layers.len() + 1);

        if let Some(first_layer) = self.layers.first() {
            dims.push(first_layer.weights[0].len());
        }

        for layer in &self.layers {
            dims.push(layer.weights.len());
        }

        dims
    }
}

impl<F, D> fmt::Debug for NeuralNetwork<F, D>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NeuralNetwork")
            .field("architecture", &self.architecture())
            .field("num_layers", &self.num_layers())
            .finish()
    }
}

fn argmax(slice: &[f64]) -> usize {
    slice
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|x| x / sum).collect()
}

fn cross_entropy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let eps = 1e-15;
    y_true.iter().zip(y_pred).fold(0.0, |acc, (&y, &p)| {
        let p = p.clamp(eps, 1.0 - eps);
        acc - y * p.ln()
    })
}

fn backprop_layer(
    downstream: &[f64],
    weights: &Matrix,
    activations: &[f64],
    activation_deriv: &impl Fn(f64) -> f64,
) -> Vec<f64> {
    (0..weights[0].len())
        .map(|i| {
            weights
                .iter()
                .zip(downstream)
                .fold(0.0, |sum, (row, &d)| sum + row[i] * d)
                * activation_deriv(activations[i])
        })
        .collect()
}

fn update_params(layer: &mut Layer, prev_activations: &[f64], deltas: &[f64], lr: f64) {
    for (neuron_idx, &delta) in deltas.iter().enumerate() {
        layer.biases[neuron_idx] -= lr * delta;
        for (w, &act) in layer.weights[neuron_idx].iter_mut().zip(prev_activations) {
            *w -= lr * delta * act;
        }
    }
}
