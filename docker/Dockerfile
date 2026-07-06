# PAL — environment-only image.
# Code and data are mounted at runtime via volumes (see docker-compose.yml).

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System packages (OpenGL/GLib for matplotlib/PIL rendering; build tools for
# any sdist that compiles; git + curl for Node/Claude Code).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 libxext6 libglib2.0-0 libgl1-mesa-glx \
    git gcc g++ python3-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Node.js (required for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Claude Code
RUN npm install -g @anthropic-ai/claude-code

# Python packages — direct deps only; pip resolves the transitive ones.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/workspace/PAL"
WORKDIR /workspace/PAL

CMD ["bash"]
