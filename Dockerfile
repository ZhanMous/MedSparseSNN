FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# 安装 LibreOffice 与 Python
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libreoffice libreoffice-core libreoffice-writer \
       python3 python3-pip ca-certificates wget unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 复制项目（在 build 时上下文应为仓库根目录）
COPY . /workspace

WORKDIR /workspace/MedSparseSNN

# 安装最小的 Python 工具（可选）
RUN pip3 install --no-cache-dir python-pptx

# 默认入口：执行容器内的转换脚本
ENTRYPOINT ["/bin/bash", "docker_convert.sh"]
