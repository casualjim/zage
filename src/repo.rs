use std::path::PathBuf;

pub fn find_repo_root(cwd: &str) -> Option<String> {
  let mut path = PathBuf::from(cwd);
  loop {
    let git_dir = path.join(".git");
    if git_dir.is_dir() || git_dir.is_file() {
      return Some(path.to_string_lossy().into_owned());
    }
    if !path.pop() {
      break;
    }
  }
  None
}
