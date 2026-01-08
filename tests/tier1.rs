use libtest_mimic::{Arguments, Trial};
use std::fs;
use std::path::{Path, PathBuf};

fn collect_toml_files(root: &Path, output: &mut Vec<PathBuf>) -> std::io::Result<()> {
  for entry in fs::read_dir(root)? {
    let entry = entry?;
    let path = entry.path();
    if path.is_dir() {
      collect_toml_files(&path, output)?;
    } else if path.extension().and_then(|ext| ext.to_str()) == Some("toml") {
      output.push(path);
    }
  }
  Ok(())
}

fn build_trials() -> Result<Vec<Trial>, Box<dyn std::error::Error>> {
  let root = PathBuf::from("src/testdata/tier1");
  let mut paths = Vec::new();
  collect_toml_files(&root, &mut paths)?;
  paths.sort();

  let mut trials = Vec::new();
  for path in paths {
    let cases = zage::predict::verifier::list_tier1_cases(&path)?;
    let rel_path = path.strip_prefix(&root).unwrap_or(&path);
    let rel_name = rel_path.to_string_lossy().replace('\\', "/");

    for case in cases {
      let name = format!("tier1::{rel_name}::{}", case.name);
      let path = path.clone();
      let index = case.index;
      trials.push(Trial::test(name, move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
          .enable_all()
          .build()
          .map_err(|err| libtest_mimic::Failed::from(err.to_string()))?;
        runtime
          .block_on(zage::predict::verifier::run_tier1_case(&path, index))
          .map_err(|err| libtest_mimic::Failed::from(err.to_string()))?;
        Ok(())
      }));
    }
  }

  Ok(trials)
}

fn main() {
  let args = Arguments::from_args();
  let trials = match build_trials() {
    Ok(trials) => trials,
    Err(err) => {
      eprintln!("failed to collect tier1 cases: {err}");
      std::process::exit(1);
    }
  };
  libtest_mimic::run(&args, trials).exit();
}
