use std::fs;
use std::path::PathBuf;
use std::process::Command;

use directories::BaseDirs;

use crate::{Result, ZageError};

const SYSTEMD_SERVICE_NAME: &str = "zage.service";
const SYSTEMD_SOCKET_NAME: &str = "zage.socket";
const LAUNCHD_LABEL: &str = "com.zage.daemon";
const LAUNCHD_SOCKET_NAME: &str = "Listeners";

pub fn install() -> Result<()> {
  if cfg!(target_os = "linux") {
    return install_systemd_user();
  }
  if cfg!(target_os = "macos") {
    return install_launchd();
  }
  Err(ZageError::ConfigError(
    "unsupported OS for service install".to_string(),
  ))
}

pub fn uninstall() -> Result<()> {
  if cfg!(target_os = "linux") {
    return uninstall_systemd_user();
  }
  if cfg!(target_os = "macos") {
    return uninstall_launchd();
  }
  Err(ZageError::ConfigError(
    "unsupported OS for service uninstall".to_string(),
  ))
}

fn zage_binary() -> Result<PathBuf> {
  std::env::current_exe().map_err(|err| ZageError::ConfigError(err.to_string()))
}

fn run_command(program: &str, args: &[&str]) -> Result<()> {
  let output = Command::new(program).args(args).output()?;
  if output.status.success() {
    return Ok(());
  }
  let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
  let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
  let args_joined = args.join(" ");
  let mut message = format!("command '{program} {args_joined}' failed");
  if !stderr.is_empty() {
    message.push_str(&format!(": {stderr}"));
  } else if !stdout.is_empty() {
    message.push_str(&format!(": {stdout}"));
  }
  Err(ZageError::ConfigError(message))
}

fn run_command_maybe(program: &str, args: &[&str]) -> Result<bool> {
  let output = Command::new(program).args(args).output()?;
  Ok(output.status.success())
}

fn user_config_dir() -> Result<PathBuf> {
  Ok(
    BaseDirs::new()
      .ok_or_else(|| ZageError::ConfigError("missing config dir".to_string()))?
      .config_dir()
      .to_path_buf(),
  )
}

fn user_home_dir() -> Result<PathBuf> {
  Ok(
    BaseDirs::new()
      .ok_or_else(|| ZageError::ConfigError("missing home dir".to_string()))?
      .home_dir()
      .to_path_buf(),
  )
}

fn launchd_socket_path() -> PathBuf {
  PathBuf::from("/tmp/zage.sock")
}

fn install_systemd_user() -> Result<()> {
  let unit_dir = user_config_dir()?.join("systemd/user");
  fs::create_dir_all(&unit_dir)?;
  let service_path = unit_dir.join(SYSTEMD_SERVICE_NAME);
  let socket_path = unit_dir.join(SYSTEMD_SOCKET_NAME);
  let zage_bin = zage_binary()?;

  let socket_content = "[Unit]\n\
Description=Zage suggestion daemon socket\n\
\n\
[Socket]\n\
ListenStream=%t/zage.sock\n\
SocketMode=0600\n\
\n\
[Install]\n\
WantedBy=sockets.target\n";
  fs::write(&socket_path, socket_content)?;

  let content = format!(
    "[Unit]\n\
Description=Zage suggestion daemon\n\
Requires={}\n\
\n\
[Service]\n\
Type=simple\n\
ExecStart={} server\n\
Environment=ZAGE_LOG=info\n\
Restart=on-failure\n\
RestartSec=5\n\
\n\
[Install]\n\
WantedBy=default.target\n",
    SYSTEMD_SOCKET_NAME,
    zage_bin.display()
  );
  fs::write(&service_path, content)?;

  run_command("systemctl", &["--user", "daemon-reload"])?;
  run_command(
    "systemctl",
    &["--user", "enable", "--now", SYSTEMD_SOCKET_NAME],
  )?;
  Ok(())
}

fn uninstall_systemd_user() -> Result<()> {
  let unit_dir = user_config_dir()?.join("systemd/user");
  let service_path = unit_dir.join(SYSTEMD_SERVICE_NAME);
  let socket_path = unit_dir.join(SYSTEMD_SOCKET_NAME);

  if service_path.exists() {
    run_command(
      "systemctl",
      &["--user", "disable", "--now", SYSTEMD_SERVICE_NAME],
    )?;
    fs::remove_file(&service_path)?;
    run_command("systemctl", &["--user", "daemon-reload"])?;
  }
  if socket_path.exists() {
    run_command(
      "systemctl",
      &["--user", "disable", "--now", SYSTEMD_SOCKET_NAME],
    )?;
    fs::remove_file(&socket_path)?;
    run_command("systemctl", &["--user", "daemon-reload"])?;
  }
  Ok(())
}

fn install_launchd() -> Result<()> {
  let home_dir = user_home_dir()?;
  let plist_dir = home_dir.join("Library/LaunchAgents");
  fs::create_dir_all(&plist_dir)?;
  let plist_path = plist_dir.join(format!("{LAUNCHD_LABEL}.plist"));
  let plist_path_str = plist_path.to_string_lossy().into_owned();
  let socket_path = launchd_socket_path();
  let zage_bin = zage_binary()?;
  let uid = uzers::get_current_uid();
  let gui_target = format!("gui/{uid}");

  let content = format!(
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\"\n\
  \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
<plist version=\"1.0\">\n\
<dict>\n\
  <key>Label</key>\n\
  <string>{}</string>\n\
  <key>ProgramArguments</key>\n\
  <array>\n\
    <string>{}</string>\n\
    <string>server</string>\n\
  </array>\n\
  <key>Sockets</key>\n\
  <dict>\n\
    <key>{}</key>\n\
    <dict>\n\
      <key>SockPathName</key>\n\
      <string>/tmp/zage.sock</string>\n\
      <key>SockPathMode</key>\n\
      <integer>384</integer>\n\
    </dict>\n\
  </dict>\n\
  <key>EnvironmentVariables</key>\n\
  <dict>\n\
    <key>ZAGE_LOG</key>\n\
    <string>info</string>\n\
  </dict>\n\
  <key>RunAtLoad</key>\n\
  <false/>\n\
</dict>\n\
</plist>\n",
    LAUNCHD_LABEL,
    zage_bin.display(),
    LAUNCHD_SOCKET_NAME
  );
  fs::write(&plist_path, content)?;

  // bootout may fail if not currently loaded; ignore in that case.
  let _ = run_command_maybe(
    "launchctl",
    &["bootout", &gui_target, plist_path_str.as_str()],
  )?;
  if socket_path.exists() {
    fs::remove_file(&socket_path)?;
  }
  if !run_command_maybe(
    "launchctl",
    &["bootstrap", &gui_target, plist_path_str.as_str()],
  )? {
    run_command("launchctl", &["load", "-w", plist_path_str.as_str()])?;
  }
  let _ = run_command_maybe(
    "launchctl",
    &["enable", &format!("{gui_target}/{LAUNCHD_LABEL}")],
  )?;
  Ok(())
}

fn uninstall_launchd() -> Result<()> {
  let home_dir = user_home_dir()?;
  let plist_path = home_dir
    .join("Library/LaunchAgents")
    .join(format!("{LAUNCHD_LABEL}.plist"));
  let plist_path_str = plist_path.to_string_lossy().into_owned();
  let socket_path = launchd_socket_path();
  let uid = uzers::get_current_uid();
  let gui_target = format!("gui/{uid}");

  if plist_path.exists() {
    let _ = run_command_maybe(
      "launchctl",
      &["bootout", &gui_target, plist_path_str.as_str()],
    )?;
    let _ = run_command_maybe("launchctl", &["unload", "-w", plist_path_str.as_str()])?;
    if socket_path.exists() {
      fs::remove_file(&socket_path)?;
    }
    fs::remove_file(&plist_path)?;
  }
  Ok(())
}
