use color_eyre::eyre::Result;

use crate::cli::ServiceAction;
use crate::service;

pub fn run(action: ServiceAction) -> Result<()> {
  match action {
    ServiceAction::Install => {
      service::install()?;
      eprintln!("Service installed");
    }
    ServiceAction::Uninstall => {
      service::uninstall()?;
      eprintln!("Service uninstalled");
    }
  }
  Ok(())
}
