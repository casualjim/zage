use std::path::{Path, PathBuf};

use crate::Result;

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

pub fn read_git_branch(repo_root: &str) -> Result<Option<String>> {
  if repo_root.is_empty() {
    return Ok(None);
  }
  let mut git_dir = PathBuf::from(repo_root).join(".git");
  if git_dir.is_file()
    && let Ok(contents) = std::fs::read_to_string(&git_dir)
    && let Some(rest) = contents.trim().strip_prefix("gitdir:")
  {
    let path = rest.trim();
    git_dir = Path::new(repo_root).join(path);
  }
  let head_path = git_dir.join("HEAD");
  let head = match std::fs::read_to_string(&head_path) {
    Ok(contents) => contents,
    Err(_) => return Ok(None),
  };
  let head = head.trim();
  if let Some(rest) = head.strip_prefix("ref:") {
    let reference = rest.trim();
    if let Some(name) = reference.rsplit('/').next()
      && !name.is_empty()
    {
      return Ok(Some(name.to_string()));
    }
  }
  Ok(None)
}
