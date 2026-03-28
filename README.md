Distributed Edge-AI Hardware-in-the-Loop (HIL) Simulation using CARLA & Jetson Nano

Overview
This project implements a Distributed Hardware-in-the-Loop (HIL) simulation framework where a CARLA simulator running on a PC streams camera frames to an NVIDIA Jetson Nano edge device for real-time AI inference. Two object detection models — SSD-MobileNet-V2 and YOLOv8 — are deployed and benchmarked on the Jetson Nano to evaluate inference performance under realistic edge conditions.
The objective is to replicate a real autonomous vehicle architecture, where sensor data is processed on an embedded edge system under realistic network conditions.

Key Features

Simulation-based testing using CARLA
Distributed edge-AI inference on Jetson Nano
Real-time camera frame streaming
TCP vs UDP communication benchmarking
Latency (RTT) measurement and analysis
Dual model support: TensorRT-optimized SSD-MobileNet-V2 and YOLOv8
Model comparison: accuracy vs latency trade-off on edge hardware

System Architecture
CARLA Simulator (PC)
│
│ RGB Camera Sensor
▼
Python CARLA Client
│
│ JPEG Encoding
▼
TCP / UDP Socket Communication
│
│ Ethernet / WiFi Network
▼
Jetson Nano (Edge Device)
│
│ TensorRT Inference
▼
AI Object Detection
(SSD-MobileNet-V2 or YOLOv8)
│
▼
Detection Results / Feedback

Communication Pipeline

Frame Transmission (PC Side)
CARLA generates RGB camera frames
Frames are converted to NumPy arrays
JPEG compression is applied
Data is transmitted via TCP/UDP socket

Packet Structure:
| Frame ID | Timestamp | Image Size | JPEG Image Data |

Frame Processing (Jetson Nano)

Receive frame via TCP/UDP

Decode JPEG image

Convert to CUDA memory

Run TensorRT inference (SSD-MobileNet-V2 or YOLOv8)

Output detection results

AI Models (Jetson Nano)
FeatureSSD-MobileNet-V2YOLOv8DatasetMS COCOMS COCOClasses9180RuntimeTensorRTTensorRTPrecisionFP16FP16Input Size300 × 300640 × 640StrengthLower latencyHigher accuracy
Both models are exported to TensorRT engine format and run natively on the Jetson Nano GPU.

Technologies Used
ComponentTechnologySimulatorCARLAEdge HardwareNVIDIA Jetson NanoAI RuntimeTensorRTModelsSSD-MobileNet-V2, YOLOv8CommunicationTCP / UDP SocketsProgrammingPythonImage ProcessingOpenCV

Key Insights

TCP achieves slightly lower steady-state latency (~52 ms) with a graceful buffer drain during warm-up
UDP shows a faster but abrupt convergence, resolving its initial backlog (~4,300 ms peak) through frame dropping rather than queuing — resulting in a 266-frame gap at start-up
Jetson inference time (~37–45 ms) is the dominant latency contributor regardless of protocol
YOLOv8 delivers higher detection accuracy at the cost of increased inference time compared to SSD-MobileNet-V2
TensorRT FP16 optimization is essential for achieving real-time performance on Jetson Nano for both models


Author
Faysal Ahammed Tonmoy
BSc Electronic Engineering — Hochschule Hamm-Lippstadt
Embedded Systems | Edge AI | Autonomous Systems
