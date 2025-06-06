use eframe::egui;

use crate::neural_network::NeuralNetwork;

pub struct DigitApp<F, D>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    canvas: [[f32; 28]; 28],
    nn: NeuralNetwork<F, D>,
    prediction: Option<Vec<f64>>,
    last_prediction_time: f64,
}

impl<F, D> DigitApp<F, D>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    pub fn new(nn: NeuralNetwork<F, D>) -> Self {
        Self {
            canvas: [[0.0; 28]; 28],
            nn,
            prediction: None,
            last_prediction_time: 0.0,
        }
    }

    fn predict(&mut self) {
        let input: Vec<f64> = self
            .canvas
            .iter()
            .flat_map(|row| row.iter().map(|&p| p as f64))
            .collect();

        let probs = self.nn.forward(&input);
        self.prediction = Some(probs);
    }
}

impl<F, D> eframe::App for DigitApp<F, D>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("Draw a digit with your mouse:");

            let response = ui.allocate_response(egui::Vec2::splat(280.0), egui::Sense::drag());
            let rect = response.rect;

            if response.dragged() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let x = ((pos.x - rect.left()) / 10.0).floor() as usize;
                    let y = ((pos.y - rect.top()) / 10.0).floor() as usize;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let nx = x + dx;
                            let ny = y + dy;
                            if nx < 28 && ny < 28 {
                                self.canvas[ny][nx] = 1.0;
                            }
                        }
                    }
                }
            }

            // Draw canvas
            let painter = ui.painter();
            for y in 0..28 {
                for x in 0..28 {
                    let val = self.canvas[y][x];
                    let color = egui::Color32::from_gray((val * 255.0) as u8);
                    let rect = egui::Rect::from_min_size(
                        rect.min + egui::vec2(x as f32 * 10.0, y as f32 * 10.0),
                        egui::vec2(10.0, 10.0),
                    );
                    painter.rect_filled(rect, 0.0, color);
                }
            }

            let now = ctx.input(|i| i.time);
            if now - self.last_prediction_time > 1.0 {
                self.predict();
                self.last_prediction_time = now;
            }

            if ui.button("Clear").clicked() {
                self.canvas = [[0.0; 28]; 28];
                self.prediction = None;
            }

            if let Some(prob) = &self.prediction {
                let mut top_predictions: Vec<(usize, f64)> =
                    prob.iter().copied().enumerate().collect();
                top_predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let (best_digit, best_prob) = top_predictions[0];
                ui.label(format!(
                    "The digit is probably {} with {:.2}% confidence.",
                    best_digit,
                    best_prob * 100.0
                ));

                for &(digit, prob) in top_predictions.iter() {
                    ui.label(format!("Digit {}: {:.2}%", digit, prob * 100.0));
                }
            }
        });
    }
}
