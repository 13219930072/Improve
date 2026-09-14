# experiment_meta — lamf

- Method: lamf
- Paper: UNVERIFIED (None)
- Official Repository: UNVERIFIED
- Commit: N/A (用户方法(Ours);论文信息由用户另行提供,此处不做推测。)
- Checkpoint: `LAMF_Full_Final_07-17-13-19.pth`
- Checkpoint SHA256: `daf4c626584ac8242d4da675123d6ac447d2292cf3309ac86c04406de4bd0720`
- Environment: Python 3.8.20 / PyTorch 1.13.1+cu117 / CUDA 11.7 / NVIDIA GeForce RTX 3090
- Dataset Root: /home/hyhz/HT/test_img (TNO 25 / RoadScene 50 / M3FD 300 / FMB 280 = 655)
- Input Order: model(vis, ir) — vis 在前(项目 test.py/train.py 惯例)
- Input Range: [0,1]
- Padding: reflect pad to 32 (H,W must be divisible by 32) + crop back
- Output Postprocess: min-max norm (+1e-8) → ×255 → uint8 → save PNG → read back float32
- Evaluator: /home/hyhz/HT/code/MY-CDDFuse/utils/Evaluator.py (Evaluator class; arg order (fused, ir, vi); SSIM data_range=255; Evaluator.VIFF not module-level)
- FLOPs Tool: thop 0.1.1 (MACs; FLOPs=2×MACs; SS2D selective_scan 自定义核不计数,见 runtime/flops.csv SpecialOpHandling 列)
- Run Command: server_eval: efficiency_benchmark + dataset_runtime + assemble_metrics(质量=eval_fusionmamba/per_image_eval 同管线 655张)
- Timestamp: 2026-09-03 19:12:32
- Status: REPRODUCED
