# 实现来源

analogy_grid.py 从当前仓库 pipeline-v6/scorers/analogy_grid.py 原样复制。仓库 git ls-files 中未找到 LICENSE 文件；本次仅在用户授权工作区内迁入，不主张取得额外再分发许可。新后端不 import 旧 pipeline。real_backend.py 根据外部 internvlu_train API 新写；固定 per-row seed、gradient accumulation、warmup/cosine 语义参考 pipeline-v6/runner/train_entry.py，但不迁入动态 Builder/probes。外部模型库 Git HEAD b8e5f356ef6a72ac8388271caa49962c9aec1994。
