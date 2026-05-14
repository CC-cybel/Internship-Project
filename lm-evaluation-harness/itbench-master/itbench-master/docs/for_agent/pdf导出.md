# PDF 报告导出 (PDF Export)

## 技能描述
此技能用于将 Markdown 格式的评估报告转换为专业的 PDF 文档。它能够自动处理中文排版、内嵌图片（如统计图表）以及表格渲染，确保报告在不同设备上的阅读体验一致且正式。

## 适用场景
- 需要向管理层或客户提交正式的评估报告时。
- 需要将报告归档或打印时。
- Markdown 文件中包含本地图片路径，直接分享 Markdown 文件会导致图片丢失时（PDF 会将图片嵌入文档中）。

## 核心工具
- **脚本路径**：`/data1/yezj/gitlab/leadbench/scripts/generate_pdf_report.py`
- **依赖库**：`reportlab` (Python)
- **字体支持**：自动检测并注册中文字体（如 WenQuanYi Zen Hei），确保中文内容不乱码。

## 输入要求
- **Markdown 文件路径**：需要转换的 `.md` 报告文件路径。
  - 例如：`/data1/yezj/gitlab/leadbench/output/comparison_report/stage2_evaluation_summary.md`
- **图片依赖**：如果 Markdown 中引用了图片（如 `![chart](chart.png)`），请确保图片文件位于 Markdown 文件的相对路径下或提供绝对路径。

## 输出结果
- 生成一个同名的 `.pdf` 文件，位于原 Markdown 文件的同一目录下（或指定输出路径）。
  - 例如：`/data1/yezj/gitlab/leadbench/output/comparison_report/stage2_evaluation_summary.pdf`

## 操作步骤

### 方法一：直接运行脚本（推荐）
如果你已经有现成的 Markdown 报告，可以直接调用脚本进行转换。

```bash
python /data1/yezj/gitlab/leadbench/scripts/generate_pdf_report.py <input_md_file> [output_pdf_file]
```

**示例**：
```bash
python /data1/yezj/gitlab/leadbench/scripts/generate_pdf_report.py \
  /data1/yezj/gitlab/leadbench/output/comparison_report/stage2_evaluation_summary.md
```

### 方法二：在代码中调用
如果需要在其他 Python 脚本中集成 PDF 生成功能，可以导入 `generate_pdf` 函数。

```python
import sys
sys.path.append('/data1/yezj/gitlab/leadbench/scripts')
from generate_pdf_report import generate_pdf

input_md = "/path/to/report.md"
output_pdf = "/path/to/report.pdf"

generate_pdf(input_md, output_pdf)
```

## 功能特性
1. **中文支持**：内置字体注册逻辑，自动寻找系统中的中文字体（如 `/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc`），解决乱码问题。
2. **智能排版**：
   - 自动识别 Markdown 标题（# H1, ## H2...）并应用相应的样式。
   - 自动解析列表（Bullet points）。
   - 自动渲染表格（Table）。
3. **图片嵌入**：自动读取 Markdown 中的图片语法 `![alt](path)`，调整图片大小以适应 A4 页面宽度，并嵌入 PDF 中。

## 常见问题
- **找不到字体**：如果提示字体注册失败，请检查系统中是否安装了 `fonts-wqy-zenhei` 或其他中文字体，或者在脚本中修改 `possible_fonts` 列表添加你的字体路径。
- **图片加载失败**：确保 Markdown 中的图片路径是正确的。如果是相对路径，脚本会基于 Markdown 文件所在的目录进行查找。
