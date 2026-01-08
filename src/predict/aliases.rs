use std::collections::HashMap;

use super::Candidate;

pub fn build_prefix_variants(prefix: &str, aliases: &HashMap<String, String>) -> Vec<String> {
  let mut variants = Vec::new();
  let trimmed = prefix.trim_start();
  if !trimmed.is_empty() {
    variants.push(trimmed.to_string());
  }
  for (alias, expansion) in aliases {
    if expansion.starts_with(trimmed) {
      variants.push(alias.clone());
    }
  }
  variants.sort();
  variants.dedup();
  variants
}

pub fn load_aliases() -> HashMap<String, String> {
  let mut map = HashMap::new();
  if let Ok(value) = std::env::var("ZAGE_ALIASES") {
    parse_aliases_into(&value, &mut map);
  }
  if let Ok(path) = std::env::var("ZAGE_ALIAS_FILE")
    && let Ok(contents) = std::fs::read_to_string(path)
  {
    parse_aliases_into(&contents, &mut map);
  }
  map
}

fn parse_aliases_into(input: &str, map: &mut HashMap<String, String>) {
  for raw in input.lines() {
    if let Some((name, value)) = parse_alias_line(raw) {
      map.insert(name, value);
    }
  }
}

fn parse_alias_line(raw: &str) -> Option<(String, String)> {
  let mut line = raw.trim();
  if line.is_empty() {
    return None;
  }
  if let Some(rest) = line.strip_prefix("alias ") {
    line = rest.trim();
  }
  let (name, value) = line.split_once('=')?;
  let name = name.split_whitespace().last().unwrap_or("").trim();
  if name.is_empty() {
    return None;
  }
  let mut value = value.trim().to_string();
  if (value.starts_with('\'') && value.ends_with('\''))
    || (value.starts_with('"') && value.ends_with('"'))
  {
    value = value[1..value.len() - 1].to_string();
  }
  if value.is_empty() {
    return None;
  }
  Some((name.to_string(), value))
}

pub(crate) fn add_alias_candidates(
  aliases: &HashMap<String, String>,
  candidates: &mut HashMap<String, Candidate>,
) {
  if aliases.is_empty() || candidates.is_empty() {
    return;
  }
  let snapshot: Vec<Candidate> = candidates.values().cloned().collect();
  let mut expansions: Vec<(String, String)> = Vec::new();
  for (alias, expansion) in aliases {
    let alias = alias.trim();
    let expansion = expansion.trim();
    if alias.is_empty() || expansion.is_empty() {
      continue;
    }
    if alias.contains(' ') {
      continue;
    }
    expansions.push((alias.to_string(), expansion.to_string()));
  }

  for candidate in snapshot {
    for (alias, expansion) in &expansions {
      if let Some(alias_command) = alias_for_command(alias, expansion, &candidate.command) {
        if candidates.contains_key(&alias_command) {
          continue;
        }
        let mut cloned = candidate.clone();
        cloned.command = alias_command.clone();
        candidates.insert(alias_command, cloned);
      }
    }
  }
}

pub(crate) fn alias_for_command(alias: &str, expansion: &str, command: &str) -> Option<String> {
  if command == expansion {
    return Some(alias.to_string());
  }
  if let Some(rest) = command.strip_prefix(expansion)
    && rest.starts_with(' ')
  {
    return Some(format!("{alias}{rest}"));
  }
  None
}

pub fn expand_alias(command: &str, aliases: &HashMap<String, String>) -> Option<String> {
  let mut parts = command.splitn(2, ' ');
  let head = parts.next()?.trim();
  let rest = parts.next().unwrap_or("");
  let expansion = aliases.get(head)?;
  if rest.is_empty() {
    return Some(expansion.to_string());
  }
  Some(format!("{expansion} {rest}"))
}

#[cfg(test)]
mod tests {
  use super::parse_alias_line;

  #[test]
  fn parse_alias_line_simple() {
    let parsed = parse_alias_line("alias ll='ls -l'");
    assert_eq!(parsed, Some(("ll".to_string(), "ls -l".to_string())));
  }

  #[test]
  fn parse_alias_line_global() {
    let parsed = parse_alias_line("alias -g G='| grep'");
    assert_eq!(parsed, Some(("G".to_string(), "| grep".to_string())));
  }

  #[test]
  fn parse_alias_line_suffix() {
    let parsed = parse_alias_line("alias -s txt='vim'");
    assert_eq!(parsed, Some(("txt".to_string(), "vim".to_string())));
  }
}
