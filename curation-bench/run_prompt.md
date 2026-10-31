---
tasks:
  - shape_color_video_ssl
iterations: 3
profile: example
runner: configs/runner.yaml
protocol: method
campaign: campaigns/example/campaign.json
---

Research video SSL supervision using a continuous next/task/build/audit/submit/finetune/eval loop. Establish the unchanged random F0-to-final baseline, then use prior code, predictions and dev feedback to test hypotheses about video selection, input/context frames, target frames and ordering. Follow the selected method protocol and frozen task contract.
