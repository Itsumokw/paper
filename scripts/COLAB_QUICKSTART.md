# SimpleMem Colab 快速运行（2-sample 分批 + 断点续跑 + 自动合并）

本指南按“最小改动”设计，复用现有脚本：

- `run_simplemem.py`
- `scripts/run_simplemem_in_batches.py`
- `scripts/merge_simplemem_results.py`

## 1) Colab 初始化

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content/drive/MyDrive
!git clone https://github.com/your-user/your-repo.git 毕业论文
%cd /content/drive/MyDrive/毕业论文
```

> 如果你已经有仓库目录，跳过 `git clone`。

## 2) 安装依赖

```bash
!python -m pip install --upgrade pip
!python -m pip install -r baseline/SimpleMem/requirements.txt
```

## 3) 配置模型（示例：GPT-5-nano）

编辑 `baseline/SimpleMem/config.py`，至少确认：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `LLM_MODEL = "gpt-5-nano"`

## 4) 分批运行（每批 2 个 sample，共 5 批）

下面命令会：

1. 自动按 `batch-size=2` 切分数据
2. 跑完每批写到独立目录
3. 全部完成后自动合并到一个 `merged/result.json`

```bash
!python scripts/run_simplemem_in_batches.py \
  --dataset datasets/locomo/data/locomo10.json \
  --batch-size 2 \
  --job-name locomo_b2_colab \
  --jobs-root /content/drive/MyDrive/simplemem_runs \
  --parallel-questions \
  --test-workers 4
```

## 5) 蓝屏/断线后续跑

继续使用同一个 `job-name` 和 `jobs-root`，即可跳过已完成批次。

```bash
!python scripts/run_simplemem_in_batches.py \
  --dataset datasets/locomo/data/locomo10.json \
  --batch-size 2 \
  --job-name locomo_b2_colab \
  --jobs-root /content/drive/MyDrive/simplemem_runs \
  --start-batch 3 \
  --parallel-questions \
  --test-workers 4
```

> 你也可以保守地每次都 `--start-batch 1`，脚本会自动跳过已有 `batch_xx/result.json`。

## 6) 结果路径

- 批次结果：`/content/drive/MyDrive/simplemem_runs/locomo_b2_colab/batch_01/result.json` ... `batch_05/result.json`
- 合并结果：`/content/drive/MyDrive/simplemem_runs/locomo_b2_colab/merged/result.json`
- 映射清单：`/content/drive/MyDrive/simplemem_runs/locomo_b2_colab/manifest.json`

## 7) 快速检查是否合并到 10 个 sample

```bash
!python - <<'PY'
import json
from pathlib import Path
p = Path('/content/drive/MyDrive/simplemem_runs/locomo_b2_colab/merged/result.json')
obj = json.loads(p.read_text(encoding='utf-8'))
print('num_samples =', obj['summary']['num_samples'])
print('num_questions =', obj['summary']['num_questions'])
print('overall F1 =', obj['aggregated_metrics']['overall']['f1']['mean'])
PY
```

