#!/bin/bash
# Cross-compilation setup script for Rust on macOS targeting Linux

# Set up environment for cross-compilation
export CC_x86_64_unknown_linux_gnu=x86_64-unknown-linux-gnu-gcc
export AR_x86_64_unknown_linux_gnu=x86_64-unknown-linux-gnu-ar
export CXX_x86_64_unknown_linux_gnu=x86_64-unknown-linux-gnu-g++

# Set up pkg-config for cross-compilation
export PKG_CONFIG_ALLOW_CROSS=1
export PKG_CONFIG_PATH_x86_64_unknown_linux_gnu=""

# Configure OpenSSL environment
export OPENSSL_STATIC=1
export OPENSSL_DIR_x86_64_unknown_linux_gnu=""

# Configure for aarch64 as well
export CC_aarch64_unknown_linux_gnu=aarch64-unknown-linux-gnu-gcc
export AR_aarch64_unknown_linux_gnu=aarch64-unknown-linux-gnu-ar
export CXX_aarch64_unknown_linux_gnu=aarch64-unknown-linux-gnu-g++

# Add Homebrew bin to PATH to ensure tools are found
export PATH="/opt/homebrew/bin:$PATH"

# Print environment variables for debugging
echo "Cross-compilation environment configured:"
echo "CC_x86_64_unknown_linux_gnu=$CC_x86_64_unknown_linux_gnu"
echo "AR_x86_64_unknown_linux_gnu=$AR_x86_64_unknown_linux_gnu"
echo "PKG_CONFIG_ALLOW_CROSS=$PKG_CONFIG_ALLOW_CROSS"
echo "OPENSSL_STATIC=$OPENSSL_STATIC"
