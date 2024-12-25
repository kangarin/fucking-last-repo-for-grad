## 项目结构

需要把yolov5的项目和权重放到对应位置，ip端口修改配置文件。

```
├── README.md
├── cloud
│   ├── display_server.py
│   └── templates
│       └── index.html
├── config.py
├── config.yaml
├── data
│   └── road_traffic.mp4  # 随便找一个视频
├── edge
│   ├── models
│   │   ├── yolov5l.pt
│   │   ├── yolov5m.pt
│   │   ├── yolov5n.pt
│   │   ├── yolov5s.pt
│   │   └── yolov5x.pt
│   ├── processing_server.py
│   └── yolov5  # yolov5的项目 git clone到这个位置
├── requirements.txt
└── stream
    └── stream_client.py
```