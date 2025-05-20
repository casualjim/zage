// //! Test client for the embedding socket server.
// //!
// //! This binary connects to the Unix domain socket server, sends embedding
// //! requests, and displays the results. It can also run benchmarks to measure
// //! the server's performance.

// use std::io::{Read, Write};
// use std::os::unix::net::UnixStream;
// use std::path::PathBuf;
// use std::time::{Duration, Instant};

// use clap::Parser;
// use zage::Result;
// use zage::ZageError;
// use zage::protocol::ProtocolMessage;
// use zage::protocol::{LengthDelimitedDecoder, LengthDelimitedEncoder};

// /// Statistics for benchmark results
// #[derive(Debug, Default)]
// struct BenchStats {
//   latencies: Vec<Duration>,
// }

// impl BenchStats {
//   /// Add a new latency measurement
//   fn add_latency(&mut self, latency: Duration) {
//     self.latencies.push(latency);
//   }

//   /// Calculate and print statistics
//   fn print_stats(&self) {
//     if self.latencies.is_empty() {
//       println!("No measurements recorded");
//       return;
//     }

//     // Sort latencies for percentile calculations
//     let mut sorted_latencies = self.latencies.clone();
//     sorted_latencies.sort();

//     // Calculate statistics
//     let min = sorted_latencies.first().unwrap();
//     let max = sorted_latencies.last().unwrap();
//     let sum: Duration = sorted_latencies.iter().sum();
//     let avg = sum / sorted_latencies.len() as u32;

//     // Calculate percentiles
//     let p90_idx = (sorted_latencies.len() as f64 * 0.9) as usize;
//     let p99_idx = (sorted_latencies.len() as f64 * 0.99) as usize;
//     let p90 = sorted_latencies.get(p90_idx).unwrap_or(max);
//     let p99 = sorted_latencies.get(p99_idx).unwrap_or(max);

//     println!("Benchmark Results:");
//     println!("  Total requests: {}", self.latencies.len());
//     println!("  Min latency:    {:?}", min);
//     println!("  Max latency:    {:?}", max);
//     println!("  Avg latency:    {:?}", avg);
//     println!("  P90 latency:    {:?}", p90);
//     println!("  P99 latency:    {:?}", p99);
//   }
// }

// /// Command line arguments for the test client
// #[derive(Parser, Debug)]
// #[clap(author, version, about, long_about = None)]
// struct Args {
//   /// Path to the Unix domain socket
//   #[clap(short, long, default_value = "/tmp/zage_embedder.sock")]
//   socket_path: PathBuf,

//   /// Text to embed
//   #[clap(short, long)]
//   text: Option<String>,

//   /// Run benchmark mode
//   #[clap(short, long)]
//   benchmark: bool,

//   /// Number of requests to send in benchmark mode
//   #[clap(short, long, default_value = "100")]
//   num_requests: usize,

//   /// Text to use for benchmark (defaults to "Hello, world!")
//   #[clap(long)]
//   bench_text: Option<String>,

//   /// Show the full embedding vector in the output
//   #[clap(short, long)]
//   verbose: bool,
// }

// /// Send an embedding request and receive the response
// fn send_embed_request(stream: &mut UnixStream, text: &str) -> Result<Vec<f32>> {
//   // Encode the request

//   // Read the response type
//   let mut type_buf = [0u8; 1];
//   stream.read_exact(&mut type_buf)?;
//   let msg_type = MessageType::from_byte(type_buf[0])?;

//   // Read the response length
//   let mut len_buf = [0u8; 4];
//   stream.read_exact(&mut len_buf)?;
//   let msg_len = u32::from_le_bytes(len_buf) as usize;

//   // Read the RLE-encoded response
//   let mut rle_buf = vec![0u8; msg_len];
//   stream.read_exact(&mut rle_buf)?;

//   // Process the response based on message type
//   match msg_type {
//     MessageType::EmbedResponse => {
//       // Decode the embedding vector
//       let mut decoder = LengthDelimitedDecoder::new(&rle_buf);
//       let embedding = decoder.decode_f32_vec()?;
//       Ok(embedding)
//     }
//     MessageType::ErrorResponse => {
//       // Decode the error message
//       let mut decoder = LengthDelimitedDecoder::new(&rle_buf);
//       let error_msg = decoder.decode_string()?;
//       Err(ZageError::ConfigError(error_msg))
//     }
//     _ => Err(ZageError::ConfigError(format!(
//       "Unexpected response type: {:?}",
//       msg_type
//     ))),
//   }
// }

// /// Run a single embedding request
// fn run_single_request(args: &Args) -> Result<()> {
//   let text = args.text.as_deref().unwrap_or("Hello, world!");
//   println!("Connecting to socket: {}", args.socket_path.display());

//   let mut stream = UnixStream::connect(&args.socket_path)?;
//   println!("Connected successfully");

//   println!("Sending embedding request for text: \"{}\"", text);
//   let start = Instant::now();
//   let embedding = send_embed_request(&mut stream, text)?;
//   let duration = start.elapsed();

//   println!(
//     "Received embedding vector with {} dimensions in {:?}",
//     embedding.len(),
//     duration
//   );

//   if args.verbose {
//     println!("Embedding vector:");
//     for (i, val) in embedding.iter().enumerate() {
//       print!("{:.6} ", val);
//       if (i + 1) % 8 == 0 {
//         println!();
//       }
//     }
//     if embedding.len() % 8 != 0 {
//       println!();
//     }
//   } else {
//     // Print just the first few values
//     println!("First 5 values: {:?}", &embedding[..5.min(embedding.len())]);
//   }

//   Ok(())
// }

// /// Run benchmark mode
// fn run_benchmark(args: &Args) -> Result<()> {
//   let text = args.bench_text.as_deref().unwrap_or("Hello, world!");
//   println!("Running benchmark with {} requests", args.num_requests);
//   println!("Text: \"{}\"", text);
//   println!("Creating a new connection for each request");

//   let mut stats = BenchStats::default();
//   let overall_start = Instant::now();

//   for i in 0..args.num_requests {
//     // Create a new connection for each request
//     let mut stream = UnixStream::connect(&args.socket_path)?;

//     let start = Instant::now();
//     let embedding = send_embed_request(&mut stream, text)?;
//     let duration = start.elapsed();
//     stats.add_latency(duration);

//     // Close the connection
//     drop(stream);

//     if (i + 1) % 10 == 0 || i == 0 {
//       println!(
//         "Completed {}/{} requests (embedding size: {})",
//         i + 1,
//         args.num_requests,
//         embedding.len()
//       );
//     }
//   }

//   let overall_duration = overall_start.elapsed();
//   println!("Benchmark completed in {:?}", overall_duration);
//   println!(
//     "Average throughput: {:.2} requests/second",
//     args.num_requests as f64 / overall_duration.as_secs_f64()
//   );

//   stats.print_stats();

//   Ok(())
// }

// fn main() -> Result<()> {
//   // Initialize logging
//   tracing_subscriber::fmt::init();

//   // Parse command line arguments
//   let args = Args::parse();

//   if args.benchmark {
//     run_benchmark(&args)
//   } else {
//     // Ensure text is provided for single request mode
//     if args.text.is_none() {
//       eprintln!("Error: --text is required when not in benchmark mode");
//       eprintln!("Use --help for usage information");
//       std::process::exit(1);
//     }
//     run_single_request(&args)
//   }
// }
fn main() {}
