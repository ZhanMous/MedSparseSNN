# 使用 Docker 在容器中生成 PPTX → PDF

在仓库根目录构建镜像：

```bash
docker build -t medsparsesnn-pptx -f Dockerfile .
```

如果仓库根目录下只有一个 .pptx 文件，直接运行：

```bash
docker run --rm -v "$(pwd):/workspace/MedSparseSNN" medsparsesnn-pptx
```

如果需要显式指定文件名，则把 PPTX 名称作为参数传入：

```bash
docker run --rm -v "$(pwd):/workspace/MedSparseSNN" medsparsesnn-pptx presentation.pptx
```

转换完成后，PDF 会输出到 outputs 目录。

注意：镜像较大（包含 LibreOffice），首次构建会较慢。
