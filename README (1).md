# Distributed Edge-AI Hardware-in-the-Loop (HIL) Simulation using CARLA & Jetson Nano

## Overview

This project implements a Distributed Hardware-in-the-Loop (HIL) simulation framework where a CARLA simulator running on a PC streams camera frames to an NVIDIA Jetson Nano edge device for real-time AI inference. Two object detection models — SSD-MobileNet-V2 and YOLOv8 — are deployed and benchmarked on the Jetson Nano to evaluate inference performance under realistic edge conditions.

The objective is to replicate a real autonomous vehicle architecture, where sensor data is processed on an embedded edge system under realistic network conditions.

---

## Key Features

- Simulation-based testing using CARLA
- Distributed edge-AI inference on Jetson Nano
- Real-time camera frame streaming
- TCP vs UDP communication benchmarking
- Latency (RTT) measurement and analysis across 200 frames
- Dual model support: TensorRT-optimized SSD-MobileNet-V2 and YOLOv8
- Model comparison: accuracy vs latency trade-off on edge hardware

---

## System Architecture

```
CARLA Simulator (PC)
│
│  RGB Camera Sensor
▼
Python CARLA Client
│
│  JPEG Encoding
▼
TCP / UDP Socket Communication
│
│  Ethernet / WiFi Network
▼
Jetson Nano (Edge Device)
│
│  TensorRT Inference
▼
AI Object Detection
(SSD-MobileNet-V2 or YOLOv8)
│
▼
Detection Results / Feedback
```

---

## Communication Pipeline

### Frame Transmission (PC Side)
- CARLA generates RGB camera frames
- Frames are converted to NumPy arrays
- JPEG compression is applied
- Data is transmitted via TCP/UDP socket

**Packet Structure:**
```
| Frame ID | Timestamp | Image Size | JPEG Image Data |
```

### Frame Processing (Jetson Nano)
- Receive frame via TCP/UDP
- Decode JPEG image
- Convert to CUDA memory
- Run TensorRT inference (SSD-MobileNet-V2 or YOLOv8)
- Output detection results

---

## AI Models (Jetson Nano)

| Feature | SSD-MobileNet-V2 | YOLOv8 |
|---|---|---|
| Dataset | MS COCO | MS COCO |
| Classes | 91 | 80 |
| Runtime | TensorRT | TensorRT |
| Precision | FP16 | FP16 |
| Input Size | 300 × 300 | 640 × 640 |
| Strength | Lower latency | Higher accuracy |

Both models are exported to TensorRT engine format and run natively on the Jetson Nano GPU.

---

## Technologies Used

| Component | Technology |
|---|---|
| Simulator | CARLA |
| Edge Hardware | NVIDIA Jetson Nano |
| AI Runtime | TensorRT |
| Models | SSD-MobileNet-V2, YOLOv8 |
| Communications | TCP / UDP Sockets |
| Programming | Python |
| Image Processing | OpenCV |

---

## Benchmark Results (200 Frames)

### Round-Trip Latency

| Metric | TCP | UDP |
|---|---|---|
| Lowest | 54.4 ms | 51.9 ms |
| Highest (peak) | 555.3 ms | 313.1 ms |
| Average | 137.7 ms | 72.2 ms |
| Median | 105.9 ms | 59.0 ms |

### Inference Time (Jetson Nano)

| Metric | TCP | UDP |
|---|---|---|
| Lowest | 35.1 ms | 35.1 ms |
| Highest | 38.4 ms | 38.4 ms |
| Average | 35.6 ms | 35.6 ms |
| Median | 35.6 ms | 35.6 ms |

> Inference time is consistent across both protocols as expected — it depends on the model and hardware, not the network.

---

## Improvement vs Previous Test (100 Frames → 200 Frames)

| Metric | TCP improvement | UDP improvement |
|---|---|---|
| Peak round-trip | −1,105.4 ms (−66.6%) | −4,014.0 ms (−92.8%) |
| Average round-trip | −201.9 ms (−59.5%) | −540.9 ms (−88.2%) |
| Consistency (std dev) | −415.5 ms (−82.5%) | −1,341.8 ms (−97.3%) |
| Peak inference time | −1,498.0 ms (−97.5%) | −4,119.6 ms (−99.1%) |

Both protocols showed dramatic improvements in peak latency and consistency. The system is now significantly more stable across a longer test run.

---

## Key Insights

- **UDP is faster on average** — UDP achieves a lower average round-trip latency (72.2 ms) compared to TCP (137.7 ms) across 200 frames, making it more suitable for high-throughput streaming scenarios
- **TCP is more consistent** — TCP shows a tighter latency distribution with a median of 105.9 ms, whereas UDP's median is 59.0 ms but with occasional spikes up to 313 ms
- **Warm-up spikes are greatly reduced** — peak round-trip dropped by 66.6% for TCP and 92.8% for UDP compared to the 100-frame test, indicating the system reaches steady state faster with longer runs
- **Inference time is stable and protocol-independent** — Jetson Nano inference averages 35.6 ms with a standard deviation of just 0.5 ms, confirming TensorRT FP16 delivers consistent real-time performance regardless of the communication protocol
- **Inference is the dominant latency contributor** — at ~35–36 ms, inference accounts for the majority of steady-state round-trip latency for both protocols
- **YOLOv8 delivers higher detection accuracy** at the cost of increased inference time compared to SSD-MobileNet-V2
- **TensorRT FP16 optimization is essential** for achieving real-time performance on Jetson Nano for both models

---

## Author

**Faysal Ahammed Tonmoy**  
BSc Electronic Engineering — Hochschule Hamm-Lippstadt  
Embedded Systems | Edge AI | Autonomous Systems
