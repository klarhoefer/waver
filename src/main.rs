use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// Input file
    #[arg(short, long)]
    input: PathBuf,

    /// Sample rate in Hz
    #[arg(short = 'r', long, default_value_t = 44100)]
    sample_rate: u32,

    /// Start position in seconds
    #[arg(short, long, default_value_t = 0.0)]
    start: f64,

    /// Duration in seconds (defaults to the rest of the input)
    #[arg(short, long)]
    duration: Option<f64>,

    /// Output .csv file for the similarities
    #[arg(short, long)]
    output: Option<PathBuf>,
}

/// Number of best matches printed to the console.
const MATCH_COUNT: usize = 5;

/// Reads a raw s16le file into a vector of samples.
fn read_samples(path: &Path) -> io::Result<Vec<i16>> {
    let bytes = fs::read(path)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect())
}

/// Duration in seconds of the given number of samples at the given sample rate.
fn duration_secs(sample_count: usize, sample_rate: u32) -> f64 {
    sample_count as f64 / sample_rate as f64
}

/// Converts a time in seconds to a sample offset at the given sample rate.
fn secs_to_offset(secs: f64, sample_rate: u32) -> usize {
    (secs * sample_rate as f64).round() as usize
}

/// A chroma vector: the log-compressed energy of the 12 pitch classes
/// (C, C#, ..., B) in one analysis frame, mean-centered and L2-normalized
/// (all zeros for silent frames).
type Chroma = [f64; 12];

/// Framing used for the STFT, derived from the sample rate.
#[derive(Clone, Copy, Debug)]
struct Framing {
    /// FFT size in samples (power of two, ~150 ms or more).
    frame_len: usize,
    /// Distance between frame starts in samples.
    hop: usize,
}

impl Framing {
    fn new(sample_rate: u32) -> Self {
        // Long enough frames are needed to resolve semitones at low pitches.
        let frame_len = ((sample_rate as f64 * 0.15) as usize).next_power_of_two();
        Framing { frame_len, hop: frame_len / 4 }
    }

    /// Number of frames that fit completely into `sample_count` samples.
    fn frame_count(&self, sample_count: usize) -> usize {
        if sample_count < self.frame_len {
            0
        } else {
            (sample_count - self.frame_len) / self.hop + 1
        }
    }
}

/// Computes one chroma vector per frame of `samples`.
///
/// Every FFT bin between ~the lowest resolvable semitone and 5 kHz adds its
/// energy to the pitch class of its frequency. Unlike the raw
/// waveform, this is insensitive to phase, small timing offsets and changes in
/// loudness or timbre, so repeated sections of a song produce similar chroma
/// sequences even when their samples differ.
fn chroma_features(samples: &[i16], sample_rate: u32, framing: Framing) -> Vec<Chroma> {
    let n = framing.frame_len;
    let fft = FftPlanner::<f64>::new().plan_fft_forward(n);
    let window: Vec<f64> = (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos())
        .collect();
    let window_sum: f64 = window.iter().sum();

    // Pitch class of every usable bin. Bins below `min_freq` are narrower than
    // the semitone spacing would require and are skipped.
    let bin_width = sample_rate as f64 / n as f64;
    let min_freq = bin_width / (2f64.powf(1.0 / 12.0) - 1.0);
    let max_freq = 5000f64.min(sample_rate as f64 / 2.0);
    let bin_class: Vec<Option<usize>> = (0..n / 2)
        .map(|k| {
            let freq = k as f64 * bin_width;
            (freq >= min_freq && freq <= max_freq).then(|| {
                let midi = 69.0 + 12.0 * (freq / 440.0).log2();
                (midi.round() as i64).rem_euclid(12) as usize
            })
        })
        .collect();

    let mut buf = vec![Complex::new(0.0, 0.0); n];
    (0..framing.frame_count(samples.len()))
        .map(|f| {
            let frame = &samples[f * framing.hop..f * framing.hop + n];
            for ((b, &x), &w) in buf.iter_mut().zip(frame).zip(&window) {
                *b = Complex::new(x as f64 / i16::MAX as f64 * w, 0.0);
            }
            fft.process(&mut buf);

            let mut chroma = [0.0; 12];
            for (b, class) in buf.iter().zip(&bin_class) {
                if let Some(c) = class {
                    chroma[*c] += (b.norm() * 2.0 / window_sum).powi(2);
                }
            }
            // Log compression relative to the strongest class, then centering,
            // so that the vector describes which pitch classes dominate.
            let max = chroma.iter().copied().fold(0.0, f64::max);
            if max < 1e-10 {
                return [0.0; 12];
            }
            chroma.iter_mut().for_each(|v| *v = (1.0 + 100.0 * *v / max).ln());
            let mean = chroma.iter().sum::<f64>() / 12.0;
            chroma.iter_mut().for_each(|v| *v -= mean);
            let norm = chroma.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-6 {
                chroma.iter_mut().for_each(|v| *v /= norm);
            } else {
                chroma = [0.0; 12];
            }
            chroma
        })
        .collect()
}

/// Matches the `reference` feature sequence against every position of
/// `features`.
///
/// Element `i` of the result is the mean cosine similarity of
/// `features[i + k]` and `reference[k]` over all `k`, i.e. the average along a
/// diagonal of the self-similarity matrix. Values are in -1.0..=1.0, where 1.0
/// means the same chroma sequence.
fn similarities(features: &[Chroma], reference: &[Chroma]) -> Vec<f64> {
    let m = reference.len();
    if m == 0 || m > features.len() {
        return Vec::new();
    }
    (0..=features.len() - m)
        .map(|i| {
            let total: f64 = features[i..i + m]
                .iter()
                .zip(reference)
                .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>())
                .sum();
            total / m as f64
        })
        .collect()
}

/// Picks up to `count` best matching positions, each at least `min_distance`
/// apart, ordered by decreasing similarity.
fn best_matches(sims: &[f64], count: usize, min_distance: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..sims.len()).collect();
    order.sort_by(|&a, &b| sims[b].total_cmp(&sims[a]));
    let mut picked: Vec<usize> = Vec::new();
    for i in order {
        if picked.len() == count {
            break;
        }
        if picked.iter().all(|&p| p.abs_diff(i) >= min_distance) {
            picked.push(i);
        }
    }
    picked
}

/// Writes the similarities as CSV with columns `time,val`, where `time` is the
/// start of the compared section in seconds.
fn write_csv(path: &Path, sims: &[f64], frame_secs: f64) -> io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "time,val")?;
    for (i, val) in sims.iter().enumerate() {
        writeln!(out, "{:.3},{val}", i as f64 * frame_secs)?;
    }
    out.flush()
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    let samples = read_samples(&args.input)?;
    let duration = duration_secs(samples.len(), args.sample_rate);
    println!("{} samples, {duration:.3} s", samples.len());

    let start = secs_to_offset(args.start, args.sample_rate);
    let end = match args.duration {
        Some(d) => secs_to_offset(args.start + d, args.sample_rate),
        None => samples.len(),
    }
    .min(samples.len());
    if start >= end {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("empty range: start offset {start}, end offset {end}"),
        ));
    }
    println!("reference: samples {start}..{end}");

    let framing = Framing::new(args.sample_rate);
    let frame_secs = duration_secs(framing.hop, args.sample_rate);
    let features = chroma_features(&samples, args.sample_rate, framing);
    let ref_first = (start as f64 / framing.hop as f64).round() as usize;
    let ref_len = ((end - start) as f64 / framing.hop as f64).round().max(1.0) as usize;
    let ref_end = (ref_first + ref_len).min(features.len());
    if ref_first >= ref_end {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "reference is too short or too close to the end of the input",
        ));
    }
    println!(
        "{} frames of {:.3} s, reference: frames {ref_first}..{ref_end}",
        features.len(),
        frame_secs
    );

    let sims = similarities(&features, &features[ref_first..ref_end]);
    println!("{} similarities computed, best matches:", sims.len());
    for i in best_matches(&sims, MATCH_COUNT, (ref_end - ref_first).div_ceil(2)) {
        println!("  {:9.3} s  {:.4}", i as f64 * frame_secs, sims[i]);
    }

    if let Some(output) = &args.output {
        write_csv(output, &sims, frame_secs)?;
        println!("wrote {}", output.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 8000;

    /// Appends `secs` seconds of a tone with the given MIDI note, amplitude
    /// and phase, plus its first overtone.
    fn push_note(samples: &mut Vec<i16>, midi: f64, secs: f64, amp: f64, phase: f64) {
        let freq = 440.0 * 2f64.powf((midi - 69.0) / 12.0);
        for i in 0..(secs * RATE as f64) as usize {
            let t = i as f64 / RATE as f64;
            let x = (2.0 * PI * freq * t + phase).sin() + 0.5 * (4.0 * PI * freq * t + phase).sin();
            samples.push((amp * x * 20000.0) as i16);
        }
    }

    #[test]
    fn finds_repetition_with_different_samples() {
        let melody = [60.0, 64.0, 67.0, 72.0, 62.0, 65.0, 69.0, 71.0];
        let other = [61.0, 66.0, 68.0, 63.0, 70.0, 61.0, 66.0, 68.0];
        let mut samples = Vec::new();
        for &m in &other {
            push_note(&mut samples, m, 0.4, 0.8, 0.0);
        }
        let first = samples.len();
        for &m in &melody {
            push_note(&mut samples, m, 0.4, 0.8, 0.0);
        }
        for &m in &other {
            push_note(&mut samples, m, 0.4, 0.8, 1.0);
        }
        // Repeat quieter, out of phase and off the frame grid; Pearson on the
        // raw samples would not recognize this.
        push_note(&mut samples, 50.0, 0.37, 0.3, 0.0);
        let second = samples.len();
        for &m in &melody {
            push_note(&mut samples, m, 0.4, 0.3, 2.0);
        }
        for &m in &other {
            push_note(&mut samples, m, 0.4, 0.3, 0.5);
        }

        let framing = Framing::new(RATE);
        let features = chroma_features(&samples, RATE, framing);
        let ref_first = first / framing.hop;
        let ref_len = (melody.len() as f64 * 0.4 * RATE as f64) as usize / framing.hop - 1;
        let sims = similarities(&features, &features[ref_first..ref_first + ref_len]);

        let matches = best_matches(&sims, 2, ref_len / 2);
        assert_eq!(matches[0], ref_first);
        assert!((sims[ref_first] - 1.0).abs() < 1e-9);
        assert!(matches[1].abs_diff(second / framing.hop) <= 1, "{matches:?}");
        assert!(sims[matches[1]] > 0.9, "{}", sims[matches[1]]);
    }

    #[test]
    fn silence_has_zero_similarity() {
        let samples = vec![0i16; 20000];
        let features = chroma_features(&samples, RATE, Framing::new(RATE));
        assert!(similarities(&features, &features[2..5]).iter().all(|&s| s == 0.0));
    }

    #[test]
    fn reference_longer_than_input_gives_empty_result() {
        assert!(similarities(&[[0.0; 12]; 2], &[[0.0; 12]; 3]).is_empty());
    }

    #[test]
    fn best_matches_keeps_distance() {
        let sims = [0.1, 0.9, 0.8, 0.2, 0.7, 0.3];
        assert_eq!(best_matches(&sims, 3, 2), vec![1, 4]);
    }
}
