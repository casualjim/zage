# Zage - The Intelligent Shell Sage

Zage (derived from "Z Shell" and "Sage") is an intelligent shell plugin that predicts the next command you're likely to run based on your shell history, working directory, and command context.

## Features

- 🔮 **Command Prediction**: Predicts the most likely next command based on your history
- 🧠 **Context Awareness**: Takes your current directory and recent commands into account
- 📊 **Sequence Learning**: Identifies and learns common command sequences
- 🔄 **Workflow Optimization**: Accelerates development and system administration tasks
- 🪄 **Zero Configuration**: Works out of the box with sensible defaults

## Installation

Build from source:

```bash
git clone https://github.com/casualjim/zage.git
cd zage
cargo build --release
```

## Quick Start

After installation, initialize Zage for your shell:

```bash
# For Zsh
echo 'eval "$(zage init zsh)"' >> ~/.zshrc

# For Bash
echo 'eval "$(zage init bash)"' >> ~/.bashrc
```

Restart your shell or source your configuration file:

```bash
source ~/.zshrc  # or ~/.bashrc
```

## Usage

Zage automatically predicts the next command you're likely to run based on your command history and current context. It seamlessly integrates with your shell workflow, without requiring manual activation.

Example workflow:

```bash
$ git pull
$ docker compose build
$ docker compose up -d
$ # Zage predicts: docker compose logs -f service-1 | humanlog
```

## How It Works

Zage uses a combination of statistical analysis and machine learning to predict commands:

1. **History Collection**: Securely stores command history with contextual metadata
2. **Pattern Recognition**: Identifies common command sequences in specific contexts
3. **LSTM Neural Network**: Learns complex patterns from your command usage
4. **Contextual Boosting**: Enhances predictions based on directory, time, and exit status

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues.

## License

MIT

## Acknowledgments

- Inspired by projects like [McFly](https://github.com/cantino/mcfly)
- Built with Rust 🦀