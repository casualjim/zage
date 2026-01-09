# Antidote/Zsh plugin entrypoint for Zage.

emulate -L zsh
local zage_root="${${(%):-%N}:A:h}"
source "${zage_root}/src/shell_integration/zsh.zsh"
