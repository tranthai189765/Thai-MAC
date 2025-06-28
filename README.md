# Thai-MAC

This project implements a multi-agent coordination algorithm based on **Hit-MAC**, designed for the **MATE** environment.

📄 Reference paper (Hit-MAC): [https://arxiv.org/pdf/2010.13110](https://arxiv.org/pdf/2010.13110)

---

## Command Usage

### Train the Coordinator
```
python changed_main.py --env Pose-v0 --model single-att --workers 6
```

### Train the Executors
```
python changed_main.py --env Pose-v1 --model multi-att-shap --workers 6
```

### Run the Trained Model
```
python changed_main.py --env Pose-v1 --model multi-att-shap --workers 0 --load-coordinator-dir trainedModel/best_coordinator.pth --load-executor-dir trainedModel/best_executor.pth
```

trainedModel/best_coordinator.pth and trainedModel/best_executor.pth are pretrained model checkpoints.
