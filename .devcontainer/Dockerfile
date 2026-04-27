FROM mcr.microsoft.com/devcontainers/base:ubuntu

USER vscode
RUN <<EOF
# https://docs.astral.sh/uv/getting-started/installation/

# Installation
curl -LsSf https://astral.sh/uv/0.11.8/install.sh | sh

# Shell autocompletion
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
echo 'eval "$(uvx --generate-shell-completion bash)"' >> ~/.bashrc
EOF