use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use dotnet_lens::search;
use serde::{Deserialize, Serialize};
use toml_edit::DocumentMut;

use crate::Result;
use crate::repo::find_repo_root;

fn find_git_root(start: &Path) -> Option<PathBuf> {
  let path = start.to_string_lossy();
  find_repo_root(path.as_ref()).map(PathBuf::from)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspacePackage {
  pub name: String,
  pub path: String,
  pub ecosystem: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceConfig {
  pub root: String,
  pub packages: Vec<WorkspacePackage>,
  pub ecosystem: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceKind {
  SingleLanguageRepo,
  Monorepo,
  PolyglotRepo,
  PolyglotMonorepo,
}

impl WorkspaceKind {
  pub fn label(&self) -> &'static str {
    match self {
      WorkspaceKind::SingleLanguageRepo => "single-language",
      WorkspaceKind::Monorepo => "monorepo",
      WorkspaceKind::PolyglotRepo => "polyglot",
      WorkspaceKind::PolyglotMonorepo => "polyglot-monorepo",
    }
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceInfo {
  pub root: String,
  pub packages: Vec<WorkspacePackage>,
  pub ecosystems: BTreeSet<String>,
  pub kind: WorkspaceKind,
}

pub fn format_workspace_summary(info: &WorkspaceInfo) -> String {
  let ecosystem = ecosystem_label(&info.ecosystems);
  let root = info.root.trim();
  let root = if root.is_empty() { "." } else { root };
  format!(
    "workspace: {} packages={} ecosystems={} root={}",
    info.kind.label(),
    info.packages.len(),
    ecosystem,
    root
  )
}

pub struct WorkspaceDetector;

impl WorkspaceDetector {
  pub fn detect(dir: &Path, files: &[String]) -> Result<Option<WorkspaceConfig>> {
    let mut configs = Self::detect_all(dir, files)?;
    Ok(configs.pop())
  }

  pub fn detect_all(dir: &Path, files: &[String]) -> Result<Vec<WorkspaceConfig>> {
    let mut configs = Vec::new();
    if let Some(config) = Self::detect_rust_workspace(dir)? {
      configs.push(config);
    }
    if let Some(config) = Self::detect_node_workspace(dir)? {
      configs.push(config);
    }
    if let Some(config) = Self::detect_go_workspace(dir)? {
      configs.push(config);
    }
    if let Some(config) = Self::detect_bazel_workspace(dir, files)? {
      configs.push(config);
    }
    if let Some(config) = Self::detect_java_workspace(dir)? {
      configs.push(config);
    }
    if let Some(config) = Self::detect_elixir_workspace(dir, files)? {
      configs.push(config);
    }
    if let Some(config) = Self::detect_deno_workspace(dir)? {
      configs.push(config);
    }
    if let Some(config) = Self::detect_haskell_workspace(dir)? {
      configs.push(config);
    }
    if let Some(config) = Self::detect_dart_workspace(dir)? {
      configs.push(config);
    }
    if let Some(config) = Self::detect_python_workspace(dir)? {
      configs.push(config);
    }
    if let Some(config) = Self::detect_ruby_workspace(dir, files)? {
      configs.push(config);
    }
    if let Some(config) = Self::detect_php_workspace(dir)? {
      configs.push(config);
    }
    if let Some(config) = Self::detect_scala_workspace(dir)? {
      configs.push(config);
    }
    if let Some(config) = Self::detect_dotnet_workspace(dir, files)? {
      configs.push(config);
    }

    Ok(configs)
  }

  fn detect_rust_workspace(dir: &Path) -> Result<Option<WorkspaceConfig>> {
    let cargo_toml = dir.join("Cargo.toml");
    if !cargo_toml.exists() {
      return Ok(None);
    }

    let content = std::fs::read_to_string(&cargo_toml)?;
    let doc = content.parse::<DocumentMut>()?;
    let workspace = doc.get("workspace");
    if workspace.is_none() {
      return Ok(None);
    }

    let members = toml_array_strings(workspace.and_then(|item| item.get("members")));
    let excludes = toml_array_strings(workspace.and_then(|item| item.get("exclude")));
    let exclude_set: HashSet<String> = excludes.into_iter().collect();
    let mut packages = Vec::new();
    for member in members {
      let member = member.trim().to_string();
      if member.is_empty() || exclude_set.contains(&member) {
        continue;
      }
      push_workspace_package(&mut packages, dir, &member, "rust");
    }

    if packages.is_empty() {
      return Ok(None);
    }

    Ok(Some(WorkspaceConfig {
      root: dir.to_string_lossy().to_string(),
      packages,
      ecosystem: "rust".to_string(),
    }))
  }

  fn detect_node_workspace(dir: &Path) -> Result<Option<WorkspaceConfig>> {
    let mut packages = Vec::new();

    let pnpm_workspace = dir.join("pnpm-workspace.yaml");
    if pnpm_workspace.exists() {
      let content = std::fs::read_to_string(&pnpm_workspace)?;
      let doc = serde_yaml::from_str::<serde_yaml::Value>(&content)?;
      if let Some(list) = doc.get("packages").and_then(|value| value.as_sequence()) {
        for entry in list {
          if let Some(pkg) = entry.as_str() {
            push_workspace_package(&mut packages, dir, pkg, "node");
          }
        }
      }
    }

    if packages.is_empty() {
      let package_json = dir.join("package.json");
      if package_json.exists() {
        let content = std::fs::read_to_string(&package_json)?;
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content)
          && let Some(workspaces) = json.get("workspaces")
        {
          let workspace_list = match workspaces {
            serde_json::Value::Array(arr) => arr.clone(),
            serde_json::Value::Object(obj) => obj
              .get("packages")
              .and_then(|packages| packages.as_array())
              .cloned()
              .unwrap_or_default(),
            _ => Vec::new(),
          };

          for ws in workspace_list {
            if let Some(ws_str) = ws.as_str() {
              push_workspace_package(&mut packages, dir, ws_str, "node");
            }
          }
        }
      }
    }

    if packages.is_empty() {
      let lerna_json = dir.join("lerna.json");
      if lerna_json.exists() {
        let content = std::fs::read_to_string(&lerna_json)?;
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content)
          && let Some(list) = json.get("packages").and_then(|value| value.as_array())
        {
          for entry in list {
            if let Some(path) = entry.as_str() {
              push_workspace_package(&mut packages, dir, path, "node");
            }
          }
        }
      }
    }

    if packages.is_empty() {
      let turbo = dir.join("turbo.json");
      let lerna = dir.join("lerna.json");
      let nx = dir.join("nx.json");
      if turbo.exists() || lerna.exists() || nx.exists() {
        for entry in std::fs::read_dir(dir)? {
          let entry = entry?;
          let path = entry.path();
          if path.is_dir() && path.join("package.json").exists() {
            let name = path
              .file_name()
              .and_then(|name| name.to_str())
              .unwrap_or("unknown")
              .to_string();
            if name != "node_modules" {
              packages.push(WorkspacePackage {
                name: name.clone(),
                path: name,
                ecosystem: "node".to_string(),
              });
            }
          }
        }
      }
    }

    if packages.is_empty() {
      return Ok(None);
    }

    Ok(Some(WorkspaceConfig {
      root: dir.to_string_lossy().to_string(),
      packages,
      ecosystem: "node".to_string(),
    }))
  }

  fn detect_go_workspace(dir: &Path) -> Result<Option<WorkspaceConfig>> {
    let go_work = dir.join("go.work");
    if !go_work.exists() {
      return Ok(None);
    }

    let content = std::fs::read_to_string(&go_work)?;
    let mut packages = Vec::new();

    for line in content.lines() {
      let trimmed = line.trim();
      if trimmed.starts_with("use") && !trimmed.starts_with("use (") {
        let path = trimmed.strip_prefix("use").unwrap_or("").trim();
        if !path.is_empty() {
          let pkg_path = dir.join(path);
          if pkg_path.exists() {
            let name = pkg_path
              .file_name()
              .and_then(|name| name.to_str())
              .unwrap_or(path)
              .to_string();
            packages.push(WorkspacePackage {
              name,
              path: path.to_string(),
              ecosystem: "go".to_string(),
            });
          }
        }
      }
    }

    if let Some(use_start) = content.find("use (") {
      let rest = &content[use_start + 5..];
      if let Some(use_end) = rest.find(')') {
        let use_block = &rest[..use_end];
        for line in use_block.lines() {
          let path = line.trim();
          if !path.is_empty() {
            let pkg_path = dir.join(path);
            if pkg_path.exists() {
              let name = pkg_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(path)
                .to_string();
              packages.push(WorkspacePackage {
                name,
                path: path.to_string(),
                ecosystem: "go".to_string(),
              });
            }
          }
        }
      }
    }

    if packages.is_empty() {
      return Ok(None);
    }

    Ok(Some(WorkspaceConfig {
      root: dir.to_string_lossy().to_string(),
      packages,
      ecosystem: "go".to_string(),
    }))
  }

  fn detect_bazel_workspace(dir: &Path, files: &[String]) -> Result<Option<WorkspaceConfig>> {
    let workspace_root = dir.join("WORKSPACE").exists()
      || dir.join("WORKSPACE.bazel").exists()
      || dir.join("MODULE.bazel").exists();
    if !workspace_root {
      return Ok(None);
    }

    let mut packages = Vec::new();
    let mut seen = HashSet::new();
    for file in files {
      let file_name = Path::new(file)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
      if file_name != "BUILD" && file_name != "BUILD.bazel" {
        continue;
      }
      let dir_path = Path::new(file).parent().unwrap_or(Path::new("."));
      let dir_str = dir_path.to_string_lossy().replace('\\', "/");
      if !seen.insert(dir_str.clone()) {
        continue;
      }
      let name = package_name_for_dir(dir, &dir_str);
      packages.push(WorkspacePackage {
        name,
        path: dir_str,
        ecosystem: "bazel".to_string(),
      });
    }

    if packages.is_empty() {
      packages.push(default_package(dir, "bazel"));
    }

    Ok(Some(WorkspaceConfig {
      root: dir.to_string_lossy().to_string(),
      packages,
      ecosystem: "bazel".to_string(),
    }))
  }

  fn detect_java_workspace(dir: &Path) -> Result<Option<WorkspaceConfig>> {
    if let Some(config) = Self::detect_gradle_workspace(dir)? {
      return Ok(Some(config));
    }
    if let Some(config) = Self::detect_maven_workspace(dir)? {
      return Ok(Some(config));
    }
    Ok(None)
  }

  fn detect_gradle_workspace(dir: &Path) -> Result<Option<WorkspaceConfig>> {
    let settings_files = ["settings.gradle", "settings.gradle.kts"];
    let mut settings_path = None;
    for name in settings_files {
      let path = dir.join(name);
      if path.exists() {
        settings_path = Some(path);
        break;
      }
    }
    let Some(settings_path) = settings_path else {
      return Ok(None);
    };
    let content = std::fs::read_to_string(&settings_path)?;
    let modules = parse_gradle_includes(&content);
    let mut packages = Vec::new();
    for module in modules {
      push_workspace_package(&mut packages, dir, &module, "gradle");
    }
    if packages.is_empty() {
      return Ok(None);
    }
    Ok(Some(WorkspaceConfig {
      root: dir.to_string_lossy().to_string(),
      packages,
      ecosystem: "gradle".to_string(),
    }))
  }

  fn detect_maven_workspace(dir: &Path) -> Result<Option<WorkspaceConfig>> {
    let pom = dir.join("pom.xml");
    if !pom.exists() {
      return Ok(None);
    }
    let content = std::fs::read_to_string(&pom)?;
    let modules = parse_maven_modules(&content);
    if modules.is_empty() {
      return Ok(None);
    }
    let mut packages = Vec::new();
    for module in modules {
      push_workspace_package(&mut packages, dir, &module, "maven");
    }
    if packages.is_empty() {
      return Ok(None);
    }
    Ok(Some(WorkspaceConfig {
      root: dir.to_string_lossy().to_string(),
      packages,
      ecosystem: "maven".to_string(),
    }))
  }

  fn detect_elixir_workspace(dir: &Path, files: &[String]) -> Result<Option<WorkspaceConfig>> {
    let mix_exs = dir.join("mix.exs");
    if !mix_exs.exists() {
      return Ok(None);
    }
    let content = std::fs::read_to_string(&mix_exs)?;
    let Some(apps_path) = parse_elixir_apps_path(&content) else {
      return Ok(None);
    };
    let mut packages = Vec::new();
    let mut seen = HashSet::new();
    for file in files {
      let path = Path::new(file);
      if path.file_name().and_then(|n| n.to_str()) != Some("mix.exs") {
        continue;
      }
      let parent = path.parent().unwrap_or(Path::new("."));
      let parent_str = parent.to_string_lossy().replace('\\', "/");
      if !parent_str.starts_with(&apps_path) {
        continue;
      }
      if !seen.insert(parent_str.clone()) {
        continue;
      }
      let name = package_name_for_dir(dir, &parent_str);
      packages.push(WorkspacePackage {
        name,
        path: parent_str,
        ecosystem: "elixir".to_string(),
      });
    }
    if packages.is_empty() {
      return Ok(None);
    }
    Ok(Some(WorkspaceConfig {
      root: dir.to_string_lossy().to_string(),
      packages,
      ecosystem: "elixir".to_string(),
    }))
  }

  fn detect_deno_workspace(dir: &Path) -> Result<Option<WorkspaceConfig>> {
    let candidates = ["deno.json", "deno.jsonc"];
    let mut config_path = None;
    for name in candidates {
      let path = dir.join(name);
      if path.exists() {
        config_path = Some(path);
        break;
      }
    }
    let Some(config_path) = config_path else {
      return Ok(None);
    };
    let content = std::fs::read_to_string(&config_path)?;
    let normalized = if config_path
      .extension()
      .and_then(|ext| ext.to_str())
      .map(|ext| ext.eq_ignore_ascii_case("jsonc"))
      .unwrap_or(false)
    {
      strip_json_comments(&content)
    } else {
      content
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&normalized) else {
      return Ok(None);
    };
    let Some(list) = json.get("workspace").and_then(|value| value.as_array()) else {
      return Ok(None);
    };
    let mut packages = Vec::new();
    for entry in list {
      if let Some(path) = entry.as_str() {
        push_workspace_package(&mut packages, dir, path, "deno");
      }
    }
    if packages.is_empty() {
      return Ok(None);
    }
    Ok(Some(WorkspaceConfig {
      root: dir.to_string_lossy().to_string(),
      packages,
      ecosystem: "deno".to_string(),
    }))
  }

  fn detect_haskell_workspace(dir: &Path) -> Result<Option<WorkspaceConfig>> {
    let cabal_project = dir.join("cabal.project");
    if !cabal_project.exists() {
      return Ok(None);
    }
    let content = std::fs::read_to_string(&cabal_project)?;
    let packages = parse_cabal_packages(&content, dir);
    if packages.is_empty() {
      return Ok(None);
    }
    Ok(Some(WorkspaceConfig {
      root: dir.to_string_lossy().to_string(),
      packages,
      ecosystem: "haskell".to_string(),
    }))
  }

  fn detect_dart_workspace(dir: &Path) -> Result<Option<WorkspaceConfig>> {
    if let Some(config) = detect_yaml_workspace(dir, "melos.yaml", "packages", "dart")? {
      return Ok(Some(config));
    }
    if let Some(config) = detect_yaml_workspace(dir, "pubspec.yaml", "workspace", "dart")? {
      return Ok(Some(config));
    }
    Ok(None)
  }

  fn detect_python_workspace(dir: &Path) -> Result<Option<WorkspaceConfig>> {
    let pyproject = dir.join("pyproject.toml");
    if !pyproject.exists() {
      return Ok(None);
    }
    let content = std::fs::read_to_string(&pyproject)?;
    let doc = content.parse::<DocumentMut>()?;
    let tool = doc.get("tool");
    let uv_workspace = tool
      .and_then(|item| item.get("uv"))
      .and_then(|item| item.get("workspace"));
    let members = toml_array_strings(uv_workspace.and_then(|item| item.get("members")));
    let excludes = toml_array_strings(uv_workspace.and_then(|item| item.get("exclude")));
    let exclude_set: HashSet<String> = excludes.into_iter().collect();
    let mut packages = Vec::new();
    for member in members {
      if exclude_set.contains(&member) {
        continue;
      }
      push_workspace_package(&mut packages, dir, &member, "python");
    }
    if packages.is_empty() {
      return Ok(None);
    }
    Ok(Some(WorkspaceConfig {
      root: dir.to_string_lossy().to_string(),
      packages,
      ecosystem: "python".to_string(),
    }))
  }

  fn detect_ruby_workspace(dir: &Path, files: &[String]) -> Result<Option<WorkspaceConfig>> {
    let mut packages = Vec::new();
    let mut seen = HashSet::new();
    for file in files {
      if !file.ends_with(".gemspec") {
        continue;
      }
      let parent = Path::new(file).parent().unwrap_or(Path::new("."));
      let parent_str = parent.to_string_lossy().replace('\\', "/");
      if !seen.insert(parent_str.clone()) {
        continue;
      }
      let name = package_name_for_dir(dir, &parent_str);
      packages.push(WorkspacePackage {
        name,
        path: parent_str,
        ecosystem: "ruby".to_string(),
      });
    }
    if packages.len() <= 1 {
      return Ok(None);
    }
    Ok(Some(WorkspaceConfig {
      root: dir.to_string_lossy().to_string(),
      packages,
      ecosystem: "ruby".to_string(),
    }))
  }

  fn detect_php_workspace(dir: &Path) -> Result<Option<WorkspaceConfig>> {
    let composer = dir.join("composer.json");
    if !composer.exists() {
      return Ok(None);
    }
    let content = std::fs::read_to_string(&composer)?;
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) else {
      return Ok(None);
    };
    let Some(repos) = json.get("repositories").and_then(|value| value.as_array()) else {
      return Ok(None);
    };
    let mut packages = Vec::new();
    for repo in repos {
      let repo_type = repo.get("type").and_then(|value| value.as_str());
      let repo_url = repo.get("url").and_then(|value| value.as_str());
      if repo_type == Some("path")
        && let Some(path) = repo_url
      {
        push_workspace_package(&mut packages, dir, path, "php");
      }
    }
    if packages.is_empty() {
      return Ok(None);
    }
    Ok(Some(WorkspaceConfig {
      root: dir.to_string_lossy().to_string(),
      packages,
      ecosystem: "php".to_string(),
    }))
  }

  fn detect_scala_workspace(dir: &Path) -> Result<Option<WorkspaceConfig>> {
    let build_sbt = dir.join("build.sbt");
    if !build_sbt.exists() {
      return Ok(None);
    }
    let content = std::fs::read_to_string(&build_sbt)?;
    let modules = parse_sbt_modules(&content);
    if modules.is_empty() {
      return Ok(None);
    }
    let mut packages = Vec::new();
    for module in modules {
      push_workspace_package(&mut packages, dir, &module, "scala");
    }
    if packages.is_empty() {
      return Ok(None);
    }
    Ok(Some(WorkspaceConfig {
      root: dir.to_string_lossy().to_string(),
      packages,
      ecosystem: "scala".to_string(),
    }))
  }

  fn detect_dotnet_workspace(dir: &Path, files: &[String]) -> Result<Option<WorkspaceConfig>> {
    let mut has_marker = false;
    let mut solution_path = None;
    for file in files {
      let lower = file.to_ascii_lowercase();
      if lower.ends_with(".sln") || lower.ends_with(".slnx") {
        has_marker = true;
        if solution_path.is_none() {
          let path = Path::new(file);
          solution_path = Some(if path.is_absolute() {
            path.to_path_buf()
          } else {
            dir.join(path)
          });
        }
      } else if lower.ends_with(".csproj")
        || lower.ends_with(".fsproj")
        || lower.ends_with(".vbproj")
      {
        has_marker = true;
      }
      if has_marker && solution_path.is_some() {
        break;
      }
    }

    if !has_marker && let Ok(entries) = std::fs::read_dir(dir) {
      for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
          continue;
        };
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".sln")
          || lower.ends_with(".slnx")
          || lower.ends_with(".csproj")
          || lower.ends_with(".fsproj")
          || lower.ends_with(".vbproj")
        {
          has_marker = true;
          if solution_path.is_none() && lower.ends_with(".sln") {
            solution_path = Some(path);
          }
          break;
        }
      }
    }

    if !has_marker {
      return Ok(None);
    }

    let mut packages = Vec::new();
    let mut seen = HashSet::new();

    if let Some(solution_path) = solution_path
      && let Ok(content) = std::fs::read_to_string(&solution_path)
    {
      let sln_dir = solution_path.parent().unwrap_or(dir);
      for project in parse_sln_projects(&content) {
        let project_path = sln_dir.join(&project);
        push_dotnet_project(&mut packages, &mut seen, dir, &project_path);
      }
    }

    if packages.is_empty() {
      let dir_buf = dir.to_path_buf();
      let projects = search::search_projects(&dir_buf)?;
      for project in projects {
        push_dotnet_project(&mut packages, &mut seen, dir, &project);
      }
    }

    if packages.is_empty() {
      return Ok(None);
    }

    Ok(Some(WorkspaceConfig {
      root: dir.to_string_lossy().to_string(),
      packages,
      ecosystem: "dotnet".to_string(),
    }))
  }

  pub fn get_package_for_file(file_path: &str, config: &WorkspaceConfig) -> Option<String> {
    let file_normalized = file_path.replace('\\', "/");
    let mut sorted_packages = config.packages.clone();
    sorted_packages.sort_by(|a, b| b.path.len().cmp(&a.path.len()));

    for package in sorted_packages {
      let pkg_normalized = package.path.replace('\\', "/");
      if file_normalized.starts_with(&pkg_normalized)
        || file_normalized.contains(&format!("/{}/", pkg_normalized))
        || file_normalized.contains(&format!("/{}", pkg_normalized))
      {
        return Some(package.name);
      }
    }

    None
  }
}

pub fn detect_workspace_info(start: &Path, files: &[String]) -> Result<WorkspaceInfo> {
  let repo_root = find_git_root(start).unwrap_or_else(|| start.to_path_buf());
  let workspace_root = find_workspace_root(&repo_root, start);
  let files_are_absolute = files.iter().any(|path| Path::new(path).is_absolute());
  let workspace_label = if files_are_absolute {
    workspace_root.clone()
  } else if let Ok(relative) = workspace_root.strip_prefix(&repo_root) {
    if relative.as_os_str().is_empty() {
      PathBuf::from(".")
    } else {
      relative.to_path_buf()
    }
  } else {
    workspace_root.clone()
  };
  let mut ecosystems = BTreeSet::new();
  let mut packages = Vec::new();
  let mut seen = HashMap::new();

  for config in WorkspaceDetector::detect_all(&workspace_root, files)? {
    ecosystems.insert(config.ecosystem.clone());
    for package in config.packages {
      let key = format!("{}:{}", package.ecosystem, package.path);
      if seen.insert(key, ()).is_some() {
        continue;
      }
      packages.push(package);
    }
  }

  let (detected_packages, detected_ecosystems) =
    detect_workspace_packages_from_files(&repo_root, files);
  for ecosystem in detected_ecosystems {
    ecosystems.insert(ecosystem);
  }
  for package in detected_packages {
    let key = format!("{}:{}", package.ecosystem, package.path);
    if seen.insert(key, ()).is_some() {
      continue;
    }
    packages.push(package);
  }

  let mut stack = BTreeSet::new();
  for file in files {
    detect_stack_from_path(Path::new(file), &mut stack);
  }
  let ecosystem = infer_ecosystem_from_stack(&stack);
  if packages.is_empty() {
    ecosystems.insert(ecosystem.clone());
    packages.push(default_package(&repo_root, &ecosystem));
  }

  Ok(build_workspace_info(workspace_label, packages, ecosystems))
}

pub fn detect_stack_from_path(path: &Path, stack: &mut BTreeSet<String>) {
  let file_name = path
    .file_name()
    .and_then(|name| name.to_str())
    .unwrap_or("")
    .to_ascii_lowercase();
  let extension = path
    .extension()
    .and_then(|ext| ext.to_str())
    .unwrap_or("")
    .to_ascii_lowercase();

  match file_name.as_str() {
    "cargo.toml" | "cargo.lock" => {
      stack.insert("Rust".to_string());
    }
    "package.json" | "pnpm-lock.yaml" | "yarn.lock" | "bun.lockb" | "bun.lock"
    | "package-lock.json" => {
      stack.insert("Node.js".to_string());
    }
    "pyproject.toml"
    | "requirements.txt"
    | "requirements-dev.txt"
    | "pipfile"
    | "poetry.lock"
    | "setup.py"
    | "setup.cfg" => {
      stack.insert("Python".to_string());
    }
    "go.mod" | "go.sum" | "go.work" => {
      stack.insert("Go".to_string());
    }
    "gemfile" | "gemfile.lock" => {
      stack.insert("Ruby".to_string());
    }
    "composer.json" | "composer.lock" => {
      stack.insert("PHP".to_string());
    }
    "pom.xml" | "mvnw" | "mvnw.cmd" => {
      stack.insert("Maven".to_string());
      stack.insert("Java".to_string());
    }
    "build.gradle"
    | "build.gradle.kts"
    | "settings.gradle"
    | "settings.gradle.kts"
    | "gradle.properties"
    | "gradlew"
    | "gradlew.bat" => {
      stack.insert("Gradle".to_string());
      stack.insert("Java".to_string());
    }
    "mix.exs" | "mix.lock" => {
      stack.insert("Elixir".to_string());
    }
    "melos.yaml" => {
      stack.insert("Melos".to_string());
    }
    "pubspec.yaml" => {
      stack.insert("Dart".to_string());
      stack.insert("Flutter".to_string());
    }
    "dockerfile" | "dockerfile.dev" => {
      stack.insert("Docker".to_string());
    }
    "deno.json" | "deno.jsonc" => {
      stack.insert("Deno".to_string());
    }
    "cabal.project" | "stack.yaml" => {
      stack.insert("Haskell".to_string());
    }
    "build.sbt" => {
      stack.insert("Scala".to_string());
    }
    "package.swift" => {
      stack.insert("Swift".to_string());
    }
    "workspace" | "workspace.bazel" | "module.bazel" | "build" | "build.bazel" => {
      stack.insert("Bazel".to_string());
    }
    _ => {}
  }

  if file_name.ends_with(".sln")
    || file_name.ends_with(".slnx")
    || file_name.ends_with(".csproj")
    || file_name.ends_with(".fsproj")
    || file_name.ends_with(".vbproj")
  {
    stack.insert("DotNet".to_string());
  }

  match extension.as_str() {
    "rs" => {
      stack.insert("Rust".to_string());
    }
    "py" => {
      stack.insert("Python".to_string());
    }
    "js" | "jsx" => {
      stack.insert("JavaScript".to_string());
    }
    "ts" | "tsx" => {
      stack.insert("TypeScript".to_string());
    }
    "go" => {
      stack.insert("Go".to_string());
    }
    "rb" => {
      stack.insert("Ruby".to_string());
    }
    "php" => {
      stack.insert("PHP".to_string());
    }
    "java" => {
      stack.insert("Java".to_string());
    }
    "kt" | "kts" => {
      stack.insert("Kotlin".to_string());
    }
    "cs" | "fs" | "vb" => {
      stack.insert(".NET".to_string());
    }
    "swift" => {
      stack.insert("Swift".to_string());
    }
    "scala" => {
      stack.insert("Scala".to_string());
    }
    "c" | "h" => {
      stack.insert("C".to_string());
    }
    "cc" | "cpp" | "cxx" | "hpp" | "hh" => {
      stack.insert("C++".to_string());
    }
    "sql" => {
      stack.insert("SQL".to_string());
    }
    "tf" | "tfvars" => {
      stack.insert("Terraform".to_string());
    }
    _ => {}
  }
}

pub fn detect_stack_from_compose(repo_root: &Path, file_path: &str, stack: &mut BTreeSet<String>) {
  let file_name = Path::new(file_path)
    .file_name()
    .and_then(|name| name.to_str())
    .unwrap_or("")
    .to_ascii_lowercase();
  if !matches_compose_file(&file_name) {
    return;
  }

  stack.insert("Docker".to_string());
  stack.insert("Docker Compose".to_string());

  let path = Path::new(file_path);
  let absolute = if path.is_absolute() {
    path.to_path_buf()
  } else {
    repo_root.join(file_path)
  };
  let content = match std::fs::read_to_string(&absolute) {
    Ok(content) => content,
    Err(_) => return,
  };
  let Ok(doc) = serde_yaml::from_str::<serde_yaml::Value>(&content) else {
    return;
  };
  let Some(services) = doc.get("services").and_then(|value| value.as_mapping()) else {
    return;
  };

  for (name, service) in services {
    if let Some(name) = name.as_str() {
      detect_dependency_from_token(name, stack);
    }
    if let Some(image) = service.get("image").and_then(|value| value.as_str()) {
      detect_dependency_from_token(image, stack);
    }
  }
}

fn detect_workspace_packages_from_files(
  repo_root: &Path,
  files: &[String],
) -> (Vec<WorkspacePackage>, BTreeSet<String>) {
  let mut packages = Vec::new();
  let mut ecosystems = BTreeSet::new();
  let mut seen = HashMap::new();

  for file in files {
    let normalized = normalize_repo_path(repo_root, file);
    let file_name = Path::new(&normalized)
      .file_name()
      .and_then(|name| name.to_str())
      .unwrap_or("")
      .to_ascii_lowercase();
    let Some(ecosystem) = stack_marker_ecosystem(&file_name) else {
      continue;
    };

    ecosystems.insert(ecosystem.to_string());
    let dir = Path::new(&normalized).parent().unwrap_or(Path::new("."));
    let dir_str = dir.to_string_lossy().replace('\\', "/");
    let dir_key = if dir_str.is_empty() {
      ".".to_string()
    } else {
      dir_str
    };
    let key = format!("{ecosystem}:{dir_key}");
    if seen.contains_key(&key) {
      continue;
    }
    seen.insert(key, ());
    let name = package_name_for_dir(repo_root, &dir_key);
    packages.push(WorkspacePackage {
      name,
      path: dir_key,
      ecosystem: ecosystem.to_string(),
    });
  }

  (packages, ecosystems)
}

fn push_workspace_package(
  packages: &mut Vec<WorkspacePackage>,
  repo_root: &Path,
  pattern: &str,
  ecosystem: &str,
) {
  for path in expand_workspace_pattern(repo_root, pattern) {
    if path.is_empty() {
      continue;
    }
    let pkg_path = repo_root.join(&path);
    if !pkg_path.exists() {
      continue;
    }
    let name = pkg_path
      .file_name()
      .and_then(|name| name.to_str())
      .unwrap_or(&path)
      .to_string();
    packages.push(WorkspacePackage {
      name,
      path,
      ecosystem: ecosystem.to_string(),
    });
  }
}

fn expand_workspace_pattern(repo_root: &Path, pattern: &str) -> Vec<String> {
  let mut pattern = pattern.trim().replace('\\', "/");
  if pattern.starts_with('!') {
    return Vec::new();
  }
  if pattern.starts_with("./") {
    pattern = pattern.trim_start_matches("./").to_string();
  }
  if pattern.is_empty() || pattern == "." {
    return vec![".".to_string()];
  }
  if !pattern.contains('*') {
    return vec![pattern];
  }

  let wildcard_index = pattern.find('*').unwrap_or(pattern.len());
  let prefix = pattern[..wildcard_index].trim_end_matches('/');
  let suffix = pattern[wildcard_index..].trim_start_matches('*');
  let base_dir = if prefix.is_empty() {
    repo_root.to_path_buf()
  } else {
    repo_root.join(prefix)
  };
  let mut expanded = Vec::new();
  let entries = match std::fs::read_dir(&base_dir) {
    Ok(entries) => entries,
    Err(_) => return expanded,
  };
  for entry in entries.flatten() {
    let path = entry.path();
    if !path.is_dir() {
      continue;
    }
    let dir_name = path
      .file_name()
      .and_then(|name| name.to_str())
      .unwrap_or("");
    if dir_name.is_empty() {
      continue;
    }
    let mut candidate = if prefix.is_empty() {
      dir_name.to_string()
    } else {
      format!("{}/{}", prefix, dir_name)
    };
    if !suffix.is_empty() {
      let trimmed = suffix.trim_start_matches('/');
      if !trimmed.is_empty() {
        candidate = format!("{}/{}", candidate, trimmed);
      }
    }
    expanded.push(candidate);
  }
  expanded
}

fn detect_yaml_workspace(
  dir: &Path,
  file_name: &str,
  key: &str,
  ecosystem: &str,
) -> Result<Option<WorkspaceConfig>> {
  let path = dir.join(file_name);
  if !path.exists() {
    return Ok(None);
  }
  let content = std::fs::read_to_string(&path)?;
  let doc = serde_yaml::from_str::<serde_yaml::Value>(&content)?;
  let mut packages = Vec::new();
  if let Some(list) = doc.get(key).and_then(|value| value.as_sequence()) {
    for entry in list {
      if let Some(path) = entry.as_str() {
        push_workspace_package(&mut packages, dir, path, ecosystem);
      }
    }
  }
  if packages.is_empty() {
    return Ok(None);
  }
  Ok(Some(WorkspaceConfig {
    root: dir.to_string_lossy().to_string(),
    packages,
    ecosystem: ecosystem.to_string(),
  }))
}

fn parse_gradle_includes(content: &str) -> Vec<String> {
  let mut modules = Vec::new();
  for line in content.lines() {
    let line = line.split("//").next().unwrap_or("").trim();
    if !line.contains("include") {
      continue;
    }
    for token in extract_quoted_strings(line) {
      let trimmed = token.trim().trim_start_matches(':');
      if trimmed.is_empty() {
        continue;
      }
      modules.push(trimmed.replace(':', "/"));
    }
  }
  modules
}

fn parse_maven_modules(content: &str) -> Vec<String> {
  let mut modules = Vec::new();
  let mut rest = content;
  while let Some(start) = rest.find("<module>") {
    let after = &rest[start + "<module>".len()..];
    if let Some(end) = after.find("</module>") {
      let value = after[..end].trim();
      if !value.is_empty() {
        modules.push(value.replace('\\', "/"));
      }
      rest = &after[end + "</module>".len()..];
    } else {
      break;
    }
  }
  modules
}

fn parse_elixir_apps_path(content: &str) -> Option<String> {
  for line in content.lines() {
    let line = line.trim();
    if !line.contains("apps_path") {
      continue;
    }
    if let Some(start) = line.find('"')
      && let Some(end) = line[start + 1..].find('"')
    {
      let value = &line[start + 1..start + 1 + end];
      return Some(value.trim().to_string());
    }
    if let Some(start) = line.find('\'')
      && let Some(end) = line[start + 1..].find('\'')
    {
      let value = &line[start + 1..start + 1 + end];
      return Some(value.trim().to_string());
    }
  }
  None
}

fn strip_json_comments(content: &str) -> String {
  let mut out = String::new();
  let mut chars = content.chars().peekable();
  while let Some(ch) = chars.next() {
    if ch == '"' {
      out.push(ch);
      let mut escaped = false;
      for next in chars.by_ref() {
        out.push(next);
        if escaped {
          escaped = false;
          continue;
        }
        if next == '\\' {
          escaped = true;
          continue;
        }
        if next == '"' {
          break;
        }
      }
      continue;
    }
    if ch == '/' {
      if let Some('/') = chars.peek().copied() {
        chars.next();
        for next in chars.by_ref() {
          if next == '\n' {
            out.push('\n');
            break;
          }
        }
        continue;
      }
      if let Some('*') = chars.peek().copied() {
        chars.next();
        while let Some(next) = chars.next() {
          if next == '*'
            && let Some('/') = chars.peek().copied()
          {
            chars.next();
            break;
          }
        }
        continue;
      }
    }
    out.push(ch);
  }
  out
}

fn parse_cabal_packages(content: &str, repo_root: &Path) -> Vec<WorkspacePackage> {
  let mut packages = Vec::new();
  let mut in_packages = false;
  for line in content.lines() {
    let trimmed = line.trim();
    if trimmed.starts_with("--") || trimmed.starts_with('#') {
      continue;
    }
    if trimmed.starts_with("packages:") {
      in_packages = true;
      let rest = trimmed.trim_start_matches("packages:").trim();
      if !rest.is_empty() {
        for entry in rest.split_whitespace() {
          push_workspace_package(&mut packages, repo_root, entry, "haskell");
        }
      }
      continue;
    }
    if in_packages {
      if trimmed.is_empty() {
        continue;
      }
      if trimmed.contains(':') {
        in_packages = false;
        continue;
      }
      for entry in trimmed.split_whitespace() {
        push_workspace_package(&mut packages, repo_root, entry, "haskell");
      }
    }
  }
  packages
}

fn parse_sbt_modules(content: &str) -> Vec<String> {
  let mut modules = Vec::new();
  for line in content.lines() {
    let trimmed = line.trim();
    if !trimmed.contains("lazy val") {
      continue;
    }
    if let Some(start) = trimmed.find("lazy val") {
      let rest = trimmed[start + "lazy val".len()..].trim();
      if let Some(name) = rest.split_whitespace().next()
        && !name.is_empty()
      {
        modules.push(name.to_string());
      }
    }
  }
  modules
}

fn parse_sln_projects(content: &str) -> Vec<String> {
  let mut projects = Vec::new();
  for line in content.lines() {
    let line = line.trim();
    if !line.starts_with("Project(") {
      continue;
    }
    let Some(rest) = line.split('=').nth(1) else {
      continue;
    };
    let mut parts = rest.split(',').map(str::trim);
    let _name = parts.next();
    let path = parts.next();
    if let Some(path) = path {
      let path = path.trim_matches('"').replace('\\', "/");
      if !path.is_empty() {
        projects.push(path);
      }
    }
  }
  projects
}

fn push_dotnet_project(
  packages: &mut Vec<WorkspacePackage>,
  seen: &mut HashSet<String>,
  root: &Path,
  project_path: &Path,
) {
  let parent = project_path.parent().unwrap_or(root);
  let relative = parent.strip_prefix(root).unwrap_or(parent);
  let mut rel = relative.to_string_lossy().replace('\\', "/");
  if rel.is_empty() {
    rel = ".".to_string();
  }
  if !seen.insert(rel.clone()) {
    return;
  }
  let name = project_path
    .file_stem()
    .and_then(|value| value.to_str())
    .map(str::to_string)
    .unwrap_or_else(|| package_name_for_dir(root, &rel));
  packages.push(WorkspacePackage {
    name,
    path: rel,
    ecosystem: "dotnet".to_string(),
  });
}

fn extract_quoted_strings(line: &str) -> Vec<String> {
  let mut values = Vec::new();
  let mut chars = line.chars().peekable();
  while let Some(ch) = chars.next() {
    if ch == '"' || ch == '\'' {
      let quote = ch;
      let mut value = String::new();
      for next in chars.by_ref() {
        if next == quote {
          break;
        }
        value.push(next);
      }
      if !value.is_empty() {
        values.push(value);
      }
    }
  }
  values
}

fn toml_array_strings(item: Option<&toml_edit::Item>) -> Vec<String> {
  let Some(item) = item else {
    return Vec::new();
  };
  let value = match item.as_value() {
    Some(value) => value,
    None => return Vec::new(),
  };
  let array = match value.as_array() {
    Some(array) => array,
    None => return Vec::new(),
  };
  array
    .iter()
    .filter_map(|entry| entry.as_str().map(|value| value.to_string()))
    .collect()
}

fn stack_marker_ecosystem(file_name: &str) -> Option<&'static str> {
  if file_name.ends_with(".sln")
    || file_name.ends_with(".slnx")
    || file_name.ends_with(".csproj")
    || file_name.ends_with(".fsproj")
    || file_name.ends_with(".vbproj")
  {
    return Some("dotnet");
  }
  if file_name.ends_with(".gemspec") {
    return Some("ruby");
  }
  match file_name {
    "cargo.toml" | "cargo.lock" => Some("rust"),
    "package.json"
    | "pnpm-lock.yaml"
    | "yarn.lock"
    | "bun.lockb"
    | "bun.lock"
    | "package-lock.json"
    | "pnpm-workspace.yaml"
    | "nx.json"
    | "turbo.json"
    | "lerna.json" => Some("node"),
    "go.mod" | "go.sum" | "go.work" => Some("go"),
    "pyproject.toml"
    | "requirements.txt"
    | "requirements-dev.txt"
    | "pipfile"
    | "poetry.lock"
    | "setup.py"
    | "setup.cfg" => Some("python"),
    "gemfile" | "gemfile.lock" => Some("ruby"),
    "composer.json" | "composer.lock" => Some("php"),
    "pom.xml" | "mvnw" | "mvnw.cmd" => Some("maven"),
    "build.gradle"
    | "build.gradle.kts"
    | "settings.gradle"
    | "settings.gradle.kts"
    | "gradle.properties"
    | "gradlew"
    | "gradlew.bat" => Some("gradle"),
    "mix.exs" | "mix.lock" => Some("elixir"),
    "deno.json" | "deno.jsonc" => Some("deno"),
    "cabal.project" | "stack.yaml" => Some("haskell"),
    "build.sbt" => Some("scala"),
    "pubspec.yaml" | "melos.yaml" => Some("dart"),
    "package.swift" => Some("swift"),
    "workspace" | "workspace.bazel" | "module.bazel" | "build" | "build.bazel" => Some("bazel"),
    _ => None,
  }
}

fn normalize_repo_path(repo_root: &Path, file_path: &str) -> String {
  let path = Path::new(file_path);
  if path.is_absolute()
    && let Ok(stripped) = path.strip_prefix(repo_root)
    && let Some(rel) = stripped.to_str()
    && !rel.is_empty()
  {
    return rel.replace('\\', "/");
  }
  file_path.replace('\\', "/")
}

fn package_name_for_dir(repo_root: &Path, dir: &str) -> String {
  if dir == "." || dir.is_empty() {
    return repo_root
      .file_name()
      .and_then(|name| name.to_str())
      .unwrap_or("repo")
      .to_string();
  }
  Path::new(dir)
    .file_name()
    .and_then(|name| name.to_str())
    .unwrap_or(dir)
    .to_string()
}

fn matches_compose_file(file_name: &str) -> bool {
  matches!(
    file_name,
    "docker-compose.yml"
      | "docker-compose.yaml"
      | "docker-compose.override.yml"
      | "docker-compose.override.yaml"
      | "compose.yml"
      | "compose.yaml"
  )
}

fn detect_dependency_from_token(token: &str, stack: &mut BTreeSet<String>) {
  let lower = token.to_ascii_lowercase();
  if lower.contains("postgres") || lower.contains("timescale") {
    stack.insert("Postgres".to_string());
  }
  if lower.contains("mysql") || lower.contains("mariadb") {
    stack.insert("MySQL".to_string());
  }
  if lower.contains("redis") {
    stack.insert("Redis".to_string());
  }
  if lower.contains("mongo") {
    stack.insert("MongoDB".to_string());
  }
  if lower.contains("kafka") {
    stack.insert("Kafka".to_string());
  }
  if lower.contains("zookeeper") {
    stack.insert("Zookeeper".to_string());
  }
  if lower.contains("rabbitmq") {
    stack.insert("RabbitMQ".to_string());
  }
  if lower.contains("nats") {
    stack.insert("NATS".to_string());
  }
  if lower.contains("elasticsearch") {
    stack.insert("Elasticsearch".to_string());
  }
  if lower.contains("opensearch") {
    stack.insert("OpenSearch".to_string());
  }
  if lower.contains("prometheus") {
    stack.insert("Prometheus".to_string());
  }
  if lower.contains("grafana") {
    stack.insert("Grafana".to_string());
  }
  if lower.contains("jaeger") {
    stack.insert("Jaeger".to_string());
  }
  if lower.contains("minio") {
    stack.insert("MinIO".to_string());
  }
  if lower.contains("clickhouse") {
    stack.insert("ClickHouse".to_string());
  }
}

pub fn stack_name_for_ecosystem(ecosystem: &str) -> Option<&'static str> {
  match ecosystem {
    "rust" => Some("Rust"),
    "go" => Some("Go"),
    "node" => Some("Node.js"),
    "python" => Some("Python"),
    "ruby" => Some("Ruby"),
    "php" => Some("PHP"),
    "java" => Some("Java"),
    "gradle" => Some("Gradle"),
    "maven" => Some("Maven"),
    "elixir" => Some("Elixir"),
    "deno" => Some("Deno"),
    "haskell" => Some("Haskell"),
    "scala" => Some("Scala"),
    "dart" => Some("Dart"),
    "bazel" => Some("Bazel"),
    "dotnet" => Some("DotNet"),
    "swift" => Some("Swift"),
    _ => None,
  }
}

pub fn build_package_map(config: &WorkspaceConfig) -> HashMap<String, String> {
  config
    .packages
    .iter()
    .map(|package| (package.path.replace('\\', "/"), package.name.clone()))
    .collect()
}

fn infer_ecosystem_from_stack(stack: &BTreeSet<String>) -> String {
  if stack.contains("Rust") {
    return "rust".to_string();
  }
  if stack.contains("Go") {
    return "go".to_string();
  }
  if stack.contains("Deno") {
    return "deno".to_string();
  }
  if stack.contains("Node.js") || stack.contains("JavaScript") || stack.contains("TypeScript") {
    return "node".to_string();
  }
  if stack.contains("Python") {
    return "python".to_string();
  }
  if stack.contains("Ruby") {
    return "ruby".to_string();
  }
  if stack.contains("PHP") {
    return "php".to_string();
  }
  if stack.contains("Gradle") {
    return "gradle".to_string();
  }
  if stack.contains("Maven") {
    return "maven".to_string();
  }
  if stack.contains("Java") {
    return "java".to_string();
  }
  if stack.contains("Elixir") {
    return "elixir".to_string();
  }
  if stack.contains("Haskell") {
    return "haskell".to_string();
  }
  if stack.contains("Scala") {
    return "scala".to_string();
  }
  if stack.contains("Flutter") || stack.contains("Dart") || stack.contains("Melos") {
    return "dart".to_string();
  }
  if stack.contains("Bazel") {
    return "bazel".to_string();
  }
  if stack.contains("DotNet") {
    return "dotnet".to_string();
  }
  if stack.contains("Swift") {
    return "swift".to_string();
  }
  "unknown".to_string()
}

fn default_package(repo_root: &Path, ecosystem: &str) -> WorkspacePackage {
  let name = repo_root
    .file_name()
    .and_then(|name| name.to_str())
    .unwrap_or("repo")
    .to_string();
  WorkspacePackage {
    name,
    path: ".".to_string(),
    ecosystem: ecosystem.to_string(),
  }
}

fn build_workspace_info(
  root: PathBuf,
  packages: Vec<WorkspacePackage>,
  ecosystems: BTreeSet<String>,
) -> WorkspaceInfo {
  let package_count = packages.len();
  let ecosystem_count = ecosystems.len();
  let is_mono = package_count > 1;
  let is_poly = ecosystem_count > 1;
  let kind = match (is_mono, is_poly) {
    (false, false) => WorkspaceKind::SingleLanguageRepo,
    (true, false) => WorkspaceKind::Monorepo,
    (false, true) => WorkspaceKind::PolyglotRepo,
    (true, true) => WorkspaceKind::PolyglotMonorepo,
  };
  WorkspaceInfo {
    root: root.to_string_lossy().to_string(),
    packages,
    ecosystems,
    kind,
  }
}

fn ecosystem_label(ecosystems: &BTreeSet<String>) -> String {
  if ecosystems.len() == 1 {
    ecosystems
      .iter()
      .next()
      .cloned()
      .unwrap_or_else(|| "unknown".to_string())
  } else {
    "multi".to_string()
  }
}

fn find_workspace_root(repo_root: &Path, start: &Path) -> PathBuf {
  let mut current = start.to_path_buf();
  loop {
    if has_workspace_marker(&current) {
      return current;
    }
    if current == repo_root {
      break;
    }
    if !current.pop() {
      break;
    }
  }
  repo_root.to_path_buf()
}

fn has_workspace_marker(dir: &Path) -> bool {
  let cargo_toml = dir.join("Cargo.toml");
  if cargo_toml.exists()
    && let Ok(content) = std::fs::read_to_string(&cargo_toml)
  {
    if let Ok(doc) = content.parse::<DocumentMut>()
      && doc.get("workspace").is_some()
    {
      return true;
    }
    if content.contains("[workspace]") {
      return true;
    }
  }

  let pnpm_workspace = dir.join("pnpm-workspace.yaml");
  if pnpm_workspace.exists() {
    return true;
  }

  let melos_yaml = dir.join("melos.yaml");
  if melos_yaml.exists() {
    return true;
  }

  let cabal_project = dir.join("cabal.project");
  let stack_yaml = dir.join("stack.yaml");
  if cabal_project.exists() || stack_yaml.exists() {
    return true;
  }

  let go_work = dir.join("go.work");
  if go_work.exists() {
    return true;
  }

  let settings_gradle = dir.join("settings.gradle");
  if settings_gradle.exists() {
    return true;
  }
  let settings_gradle_kts = dir.join("settings.gradle.kts");
  if settings_gradle_kts.exists() {
    return true;
  }

  let pom = dir.join("pom.xml");
  if pom.exists()
    && let Ok(content) = std::fs::read_to_string(&pom)
    && content.contains("<modules>")
  {
    return true;
  }

  let turbo = dir.join("turbo.json");
  if turbo.exists() {
    return true;
  }

  let lerna = dir.join("lerna.json");
  if lerna.exists() {
    return true;
  }

  let nx = dir.join("nx.json");
  if nx.exists() {
    return true;
  }

  let package_json = dir.join("package.json");
  if package_json.exists()
    && let Ok(content) = std::fs::read_to_string(&package_json)
    && let Ok(json) = serde_json::from_str::<serde_json::Value>(&content)
    && json.get("workspaces").is_some()
  {
    return true;
  }

  for name in ["deno.json", "deno.jsonc"] {
    let path = dir.join(name);
    if path.exists()
      && let Ok(content) = std::fs::read_to_string(&path)
    {
      let normalized = if name.ends_with("jsonc") {
        strip_json_comments(&content)
      } else {
        content
      };
      if let Ok(json) = serde_json::from_str::<serde_json::Value>(&normalized)
        && json.get("workspace").is_some()
      {
        return true;
      }
    }
  }

  let pyproject = dir.join("pyproject.toml");
  if pyproject.exists()
    && let Ok(content) = std::fs::read_to_string(&pyproject)
    && let Ok(doc) = content.parse::<DocumentMut>()
  {
    let tool = doc.get("tool");
    if tool
      .and_then(|item| item.get("uv"))
      .and_then(|item| item.get("workspace"))
      .is_some()
    {
      return true;
    }
  }

  let pubspec = dir.join("pubspec.yaml");
  if pubspec.exists()
    && let Ok(content) = std::fs::read_to_string(&pubspec)
    && let Ok(doc) = serde_yaml::from_str::<serde_yaml::Value>(&content)
    && doc.get("workspace").is_some()
  {
    return true;
  }

  let mix_exs = dir.join("mix.exs");
  if mix_exs.exists()
    && let Ok(content) = std::fs::read_to_string(&mix_exs)
    && parse_elixir_apps_path(&content).is_some()
  {
    return true;
  }

  if dir.join("WORKSPACE").exists()
    || dir.join("WORKSPACE.bazel").exists()
    || dir.join("MODULE.bazel").exists()
  {
    return true;
  }

  if let Ok(entries) = std::fs::read_dir(dir) {
    for entry in entries.flatten() {
      let path = entry.path();
      if matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("sln") | Some("slnx") | Some("csproj") | Some("fsproj") | Some("vbproj")
      ) {
        return true;
      }
    }
  }

  false
}

pub fn detect_workspace_for_cwd(cwd: &str) -> Result<Option<WorkspaceInfo>> {
  let path = Path::new(cwd);
  if !path.exists() {
    return Ok(None);
  }
  let files: Vec<String> = Vec::new();
  let info = detect_workspace_info(path, &files)?;
  Ok(Some(info))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_package_map() {
    let config = WorkspaceConfig {
      root: "/test".to_string(),
      packages: vec![WorkspacePackage {
        name: "core".to_string(),
        path: "packages/core".to_string(),
        ecosystem: "node".to_string(),
      }],
      ecosystem: "node".to_string(),
    };

    let map = build_package_map(&config);
    assert_eq!(map.get("packages/core"), Some(&"core".to_string()));
  }
}
