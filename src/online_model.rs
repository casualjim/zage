use crate::core::Invocation;

fn normalize_workspace_root_value(root: &str) -> String {
  let trimmed = root.trim_end_matches('/');
  if trimmed.is_empty() {
    return String::new();
  }
  std::path::Path::new(trimmed)
    .file_name()
    .map(|v| v.to_string_lossy().into_owned())
    .unwrap_or_else(|| trimmed.to_string())
}

fn normalize_cwd_value(cwd: &str, workspace_root: Option<&str>) -> String {
  let trimmed = cwd.trim_end_matches('/');
  if trimmed.is_empty() {
    return String::new();
  }

  if let Some(root) = workspace_root.filter(|v| !v.is_empty()) {
    let root_trimmed = root.trim_end_matches('/');
    if !root_trimmed.is_empty() {
      let cwd_path = std::path::Path::new(trimmed);
      let root_path = std::path::Path::new(root_trimmed);
      if let Ok(rel) = cwd_path.strip_prefix(root_path) {
        let rel_str = rel.to_string_lossy();
        if !rel_str.trim().is_empty() {
          return rel_str.into_owned();
        }
      }
    }
  }

  std::path::Path::new(trimmed)
    .file_name()
    .map(|v| v.to_string_lossy().into_owned())
    .unwrap_or_else(|| trimmed.to_string())
}

pub(crate) mod replay;
pub(crate) mod sampler;
pub(crate) mod trainer;

#[derive(Debug, Clone, Copy)]
pub struct OnlineContextInput<'a> {
  pub workspace_root: Option<&'a str>,
  pub cwd: Option<&'a str>,
  pub hostname: Option<&'a str>,
  pub username: Option<&'a str>,
  pub git_branch: Option<&'a str>,
  pub exit_status: Option<i64>,
  pub session_id: Option<i64>,
  pub unix_timestamp: Option<i64>,
}

pub fn context_tokens(input: OnlineContextInput<'_>) -> Vec<String> {
  let mut out = Vec::new();

  if let Some(root) = input.workspace_root.filter(|v| !v.is_empty()) {
    let normalized = normalize_workspace_root_value(root);
    out.push(format!("ctx:workspace_root={normalized}"));
  }
  if let Some(cwd) = input.cwd.filter(|v| !v.is_empty()) {
    let normalized = normalize_cwd_value(cwd, input.workspace_root);
    out.push(format!("ctx:cwd={normalized}"));
  }
  if let Some(exit) = input.exit_status {
    out.push(format!("ctx:exit={exit}"));
  }
  if let Some(host) = input.hostname.filter(|v| !v.is_empty()) {
    out.push(format!("ctx:host={host}"));
  }
  if let Some(user) = input.username.filter(|v| !v.is_empty()) {
    out.push(format!("ctx:user={user}"));
  }
  if let Some(branch) = input.git_branch.filter(|v| !v.is_empty()) {
    out.push(format!("ctx:git_branch={branch}"));
  }
  if let Some(ts) = input.unix_timestamp {
    out.push(format!("ctx:timebucket={}", time_bucket(ts)));
  }
  if let Some(session) = input.session_id {
    out.push(format!("ctx:session={session}"));
  }

  out
}

pub fn context_tokens_from_invocation(inv: &Invocation) -> Vec<String> {
  let workspace_root = inv.workspace.as_ref().map(|w| w.root.as_str());
  let git_branch = inv.workspace.as_ref().and_then(|w| w.git_branch.as_deref());
  let cwd = inv.working_directory.as_deref();
  let hostname = inv.hostname.as_deref();
  let username = inv.username.as_deref();
  let exit_status = inv.exit_status;
  let session_id = Some(inv.session_id);
  let unix_timestamp = inv.end_unix_timestamp.or(inv.start_unix_timestamp);

  context_tokens(OnlineContextInput {
    workspace_root,
    cwd,
    hostname,
    username,
    git_branch,
    exit_status,
    session_id,
    unix_timestamp,
  })
}

pub fn command_tokens(shellname: &str, command: &str) -> Vec<String> {
  crate::tokenize::generalized_command_tokens(shellname, command, 8)
}

pub fn window_tokens(shellname: &str, recent_commands: &[String], window: usize) -> Vec<String> {
  if window == 0 || recent_commands.is_empty() {
    return Vec::new();
  }

  let mut out = Vec::new();
  for (age, cmd) in recent_commands.iter().rev().take(window).enumerate() {
    let age = age + 1;
    for tok in command_tokens(shellname, cmd) {
      out.push(format!("prev{age}:{tok}"));
    }
  }
  out
}

pub fn subword_buckets_for_token(token: &str) -> Vec<(u32, f32)> {
  let bucket_count = crate::config::OnlineModelConfig::load()
    .map(|cfg| cfg.bucket_count)
    .unwrap_or(crate::hash_util::SUBWORD_BUCKETS);
  let mut scratch_indices = Vec::new();
  let mut scratch = Vec::new();
  crate::hash_util::stable_char_ngrams_buckets(
    token,
    bucket_count,
    &mut scratch_indices,
    &mut scratch,
  );
  scratch
}

pub fn subword_buckets_for_tokens(tokens: &[String]) -> Vec<(u32, f32)> {
  let bucket_count = crate::config::OnlineModelConfig::load()
    .map(|cfg| cfg.bucket_count)
    .unwrap_or(crate::hash_util::SUBWORD_BUCKETS);
  let mut out = Vec::new();
  let mut scratch_indices = Vec::new();
  let mut scratch = Vec::new();
  for token in tokens {
    crate::hash_util::stable_char_ngrams_buckets(
      token,
      bucket_count,
      &mut scratch_indices,
      &mut scratch,
    );
    out.extend(scratch.iter().copied());
  }
  out
}

fn time_bucket(ts: i64) -> u8 {
  if ts <= 0 {
    return 0;
  }
  let hour = ((ts / 3600) % 24) as u8;
  match hour {
    0..=5 => 1,
    6..=11 => 2,
    12..=17 => 3,
    _ => 4,
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::hash_util::{stable_bucket_and_sign, stable_char_ngrams_buckets};

  #[test]
  fn generalized_command_tokens_flags_are_sorted_and_deduped() {
    let tokens = crate::tokenize::generalized_command_tokens("zsh", "git commit -m a -a -m b", 8);
    assert_eq!(
      tokens,
      vec![
        "head:git",
        "flag:-a",
        "flag:-m",
        "arg:commit",
        "arg:a",
        "arg:b"
      ]
    );
  }

  #[test]
  fn generalized_command_tokens_args_are_normalized_and_bounded() {
    let tokens = crate::tokenize::generalized_command_tokens(
      "zsh",
      "curl https://example.com 1234ABCD /tmp/file FooBar 9 extra",
      4,
    );
    assert_eq!(
      tokens,
      vec![
        "head:curl",
        "arg:PATH",
        "arg:HASH",
        "arg:PATH",
        "arg:foobar"
      ]
    );
  }

  #[test]
  fn generalized_command_tokens_golden_cases() {
    let cases: Vec<(&str, &str, usize, Vec<&str>)> = vec![
      (
        "zsh",
        "git status -sb --porcelain",
        8,
        vec!["head:git", "flag:--porcelain", "flag:-sb", "arg:status"],
      ),
      (
        "zsh",
        "rm -- -rf /tmp/FILE",
        8,
        vec!["head:rm", "arg:-rf", "arg:PATH"],
      ),
      (
        "zsh",
        "python -m http.server 8000",
        8,
        vec!["head:python", "flag:-m", "arg:http.server", "arg:NUM"],
      ),
    ];

    for (shellname, input, max_args, expected) in cases {
      let got = crate::tokenize::generalized_command_tokens(shellname, input, max_args);
      let expected = expected
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
      assert_eq!(got, expected, "input: {input:?}");
    }
  }

  #[test]
  fn window_tokens_prefix_by_age() {
    let recent = vec!["git status -sb".to_string(), "ls -la /tmp/file".to_string()];
    let tokens = window_tokens("zsh", &recent, 10);
    assert!(!tokens.is_empty());
    assert!(tokens.iter().any(|t| t.starts_with("prev1:")));
    assert!(tokens.iter().any(|t| t.starts_with("prev2:")));
  }

  #[test]
  fn context_tokens_include_timebucket_when_timestamp_is_present() {
    let tokens = context_tokens(OnlineContextInput {
      workspace_root: Some("/workspace"),
      cwd: Some("/workspace/crate"),
      hostname: Some("host"),
      username: Some("user"),
      git_branch: None,
      exit_status: Some(2),
      session_id: Some(42),
      unix_timestamp: Some(1_700_000_000),
    });
    assert!(tokens.iter().any(|t| t.starts_with("ctx:timebucket=")));
    assert!(tokens.contains(&"ctx:workspace_root=workspace".to_string()));
    assert!(tokens.contains(&"ctx:cwd=crate".to_string()));
  }

  #[test]
  fn stable_bucket_and_sign_is_deterministic() {
    let bucket_count = crate::config::OnlineModelConfig::default().bucket_count;
    let (b1, s1) = stable_bucket_and_sign("tok", bucket_count);
    let (b2, s2) = stable_bucket_and_sign("tok", bucket_count);
    assert_eq!(b1, b2);
    assert_eq!(s1, s2);
  }

  #[test]
  fn stable_char_ngram_hashing_is_deterministic_and_in_range() {
    let bucket_count = crate::config::OnlineModelConfig::default().bucket_count;
    let mut idx1 = Vec::new();
    let mut out1 = Vec::new();
    stable_char_ngrams_buckets("token", bucket_count, &mut idx1, &mut out1);
    let mut idx2 = Vec::new();
    let mut out2 = Vec::new();
    stable_char_ngrams_buckets("token", bucket_count, &mut idx2, &mut out2);
    assert_eq!(out1, out2);
    assert!(out1.iter().all(|(b, _)| *b < bucket_count));
  }
}
