use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::Parser;

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

/// Number of similarity values combined into one CSV line.
const CSV_CHUNK_SIZE: usize = 60;

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

/// Computes the normalized cross-correlation (Pearson coefficient) between
/// `reference` and every window of the same length in `samples`.
///
/// Element `i` of the result is the similarity of `samples[i..i + reference.len()]`
/// with `reference`, in the range -1.0..=1.0. Windows or references with zero
/// variance (e.g. silence) get a similarity of 0.0.
fn similarities(samples: &[i16], reference: &[i16]) -> Vec<f64> {
    let m = reference.len();
    if m == 0 || m > samples.len() {
        return Vec::new();
    }

    // Center the reference so that dot(window, ref_centered) equals the
    // covariance sum without having to subtract the window mean.
    let ref_mean = reference.iter().map(|&y| y as f64).sum::<f64>() / m as f64;
    let ref_centered: Vec<f64> = reference.iter().map(|&y| y as f64 - ref_mean).collect();
    let ref_norm = ref_centered.iter().map(|y| y * y).sum::<f64>().sqrt();

    // Prefix sums of x and x² give each window's variance in O(1).
    let mut sum = vec![0i64; samples.len() + 1];
    let mut sum_sq = vec![0i64; samples.len() + 1];
    for (i, &x) in samples.iter().enumerate() {
        let x = x as i64;
        sum[i + 1] = sum[i] + x;
        sum_sq[i + 1] = sum_sq[i] + x * x;
    }

    (0..=samples.len() - m)
        .map(|i| {
            let s = (sum[i + m] - sum[i]) as f64;
            let s_sq = (sum_sq[i + m] - sum_sq[i]) as f64;
            let window_norm = (s_sq - s * s / m as f64).max(0.0).sqrt();
            let denom = window_norm * ref_norm;
            if denom == 0.0 {
                return 0.0;
            }
            let dot: f64 = samples[i..i + m]
                .iter()
                .zip(&ref_centered)
                .map(|(&x, &y)| x as f64 * y)
                .sum();
            dot / denom
        })
        .collect()
}

/// Writes the similarities as CSV with columns `pos,val`, where `pos` is the
/// line number (starting at 0) and `val` is the maximum of each chunk of
/// `chunk_size` consecutive similarities.
fn write_csv(path: &Path, sims: &[f64], chunk_size: usize) -> io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "pos,val")?;
    for (pos, chunk) in sims.chunks(chunk_size).enumerate() {
        let val = chunk.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        writeln!(out, "{pos},{val}")?;
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

    let sims = similarities(&samples, &samples[start..end]);
    println!("{} similarities computed", sims.len());
    println!("{:?}", &sims[..sims.len().min(10)]);

    if let Some(output) = &args.output {
        write_csv(output, &sims, CSV_CHUNK_SIZE)?;
        println!("wrote {}", output.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_repeated_pattern() {
        let pattern: Vec<i16> = (0..100).map(|i| ((i * 37) % 200 - 100) as i16 * 50).collect();
        let mut samples: Vec<i16> = (0..1000).map(|i| ((i * 7919) % 401 - 200) as i16).collect();
        samples[200..300].copy_from_slice(&pattern);
        samples[700..800].copy_from_slice(&pattern);

        let sims = similarities(&samples, &samples[200..300]);
        assert_eq!(sims.len(), samples.len() - 100 + 1);
        assert!((sims[200] - 1.0).abs() < 1e-9);
        assert!((sims[700] - 1.0).abs() < 1e-9);
        assert!(sims.iter().enumerate().all(|(i, &s)| i == 200 || i == 700 || s < 0.99));
    }

    #[test]
    fn silence_has_zero_similarity() {
        let samples = vec![0i16; 50];
        assert!(similarities(&samples, &samples[10..20]).iter().all(|&s| s == 0.0));
    }

    #[test]
    fn reference_longer_than_input_gives_empty_result() {
        assert!(similarities(&[1, 2], &[1, 2, 3]).is_empty());
    }
}
