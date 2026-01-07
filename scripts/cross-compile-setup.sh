#!/bin/bash
# Cross-compilation setup script for Rust using cargo-zigbuild + Zig

# Add Homebrew bin to PATH to ensure tools are found on macOS.
export PATH="/opt/homebrew/bin:$PATH"

# Configure OpenSSL environment for cross builds (if needed by dependencies).
export PKG_CONFIG_ALLOW_CROSS=1
export OPENSSL_STATIC=1

echo "Cross-compilation environment configured for cargo-zigbuild:"
echo "PATH=$PATH"
echo "PKG_CONFIG_ALLOW_CROSS=$PKG_CONFIG_ALLOW_CROSS"
echo "OPENSSL_STATIC=$OPENSSL_STATIC"
echo
echo "Next steps:"
echo "  - Install Zig: brew install zig"
echo "  - Install cargo-zigbuild: cargo install --locked cargo-zigbuild"
echo "  - Build: cargo zigbuild --release --target x86_64-unknown-linux-gnu"
echo "  - Build: cargo zigbuild --release --target aarch64-unknown-linux-gnu"
