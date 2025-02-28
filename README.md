# Thai-MAC
Thuật toán được xây dựng dựa trên thuật toán Hit-MAC cho môi trường MATE
Thuật toán Hit-MAC : https://arxiv.org/pdf/2010.13110

# Cấu trúc các câu lệnh
Để train coordinator: 
```
python changed_main.py --env Pose-v0 --model single-att --workers 6
```

Để train executores : 
```
python changed_main.py --env Pose-v1 --model multi-att-shap --workers 6
```

Để chạy model : 
```
python changed_main.py --env Pose-v1 --model multi-att-shap --workers 0 --load-coordinator-dir trainedModel/best_coordinator.pth --load-executor-dir trainedModel/best_executor.pth
```

Trong đó trainedModel/best_coordinator.pth và trainedModel/best_executor.pth là các models đã được train
