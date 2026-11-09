# Steer-DrivoR 数据审计材料

本仓库收录 2026-11-09 对第一阶段定位文本和第二阶段带导航文本的实际缓存审计。用途是交给另一位研究者或 AI，核查数据生成流程、统计、图文对应和接地方案。这里是固定快照，不是完整 NAVSIM 数据集，也不是已完成的训练工程。

## 建议阅读顺序

1. [给另一个 AI 的任务说明](AI_REVIEW_REQUEST.md)
2. [完整文字报告](report.md)：逐项回答八类问题，包含每组 3 条完整原始样本。
3. [单文件可视化报告](report.html)：下载后用浏览器打开，内嵌 10 张四视图拼图和一张前视原图。
4. [验证结果](verification.json)、[划分重叠检查](split_overlap.json)、[生成进度快照](audit_manifest.json)。

## 快照中的关键事实

| 项目 | 第一阶段 | 第二阶段（最新 rerun） |
|---|---:|---:|
| 选中场景 | 85,109 | 85,109 |
| 已读取文本缓存 | 65,311 | 9,200 |
| 选中 logs | 978 | 978 |
| 与本工程 navtest log/token 重叠 | 0 / 0 | 0 / 0 |

两组使用同一批 navtrain/train 原始场景，区别在文本生成策略，不是互斥的训练/测试划分。生成仍未完成，最终全量投影标签与索引尚未合并；本工程也尚未接入四相机文本训练 Dataset。完整口径和限制见报告，不要将 prepared 场景数当作可训练成品数。

## 文件导航

| 文件或目录 | 内容 |
|---|---|
| `report.md` / `report.html` | 完整报告；HTML 包含图片，可离线单文件查看 |
| `samples/stage1/`、`samples/stage2/` | 原样缓存及对应 manifest record；含报告样本与可视化样本 |
| `raw_outputs/` | 20 条未经 JSON 解析的已保存 VLM 输出及来源索引 |
| `images/` | 10 张拼图，顺序 f0、b0、l0、r0；`index.json` 包含文字和原图路径 |
| `calibration/` | 一个场景的原始标定、GT 数值、数组、前视原图 |
| `stage1_statistics.json`、`stage2_statistics.json` | 按相机统计、空文本原因、短语数和类别分布 |
| `*_snapshot.csv` | 本次读取文件清单及 SHA-256；不是额外的场景数据 |
| `*_selected_logs.txt`、`*_cached_logs.txt` | 完整选中/已缓存 log 名列表 |
| `reference_*_logs.txt` | 本工程 navtest 和 navtrain/val 对照 log 名列表 |
| `metadata/` | 每个 rank 的生成配置、完整 prompt、模型/源码指纹 |
| `partition_status.json` | 全量与 pilot 的 train/val/test 完成状态 |
| `code/` | 实际代码快照、原指导文档、审计脚本；`index.json` 提供来源与 SHA-256 |

原始 JSON、源码和报告保留了原环境绝对路径。这些是来源证据，在另一台机器上通常不可直接访问；阅读附件请使用仓库内相对路径。除已附的原始前视图和拼图外，其余原始图片、模型权重及全量缓存不在仓库中。

`code/` 是审核用的源码副本，不是可直接启动训练的完整工程。无需下载大模型或重新生成文本就能阅读报告、检查样本和基础统计。

## 提供给另一个 AI

将仓库地址和 `AI_REVIEW_REQUEST.md` 的内容一起发给对方。如果仓库私有，需通过对方支持的仓库连接器授权，或直接上传本地材料压缩包。仅提供私有仓库 URL 不会自动赋予访问权限。

## 上传到你自己的远端

本地已使用独立 Git 仓库保存材料。先在 GitHub、GitLab 或 Gitee 建立一个空仓库，然后在本仓库目录执行：

```bash
bash scripts/push_to_remote.sh git@github.com:YOUR_ACCOUNT/steer-drivor-data-audit.git
```

替换成自己的实际仓库地址；使用已配置的 SSH 或 Git 凭据。不要把密码/令牌写入仓库。脚本不强制推送，也不覆盖不同的已有 origin。
