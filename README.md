Distributed Edge-AI Hardware-in-the-Loop (HIL) Simulation using CARLA & Jetson Nano
Overview

This project implements a Distributed Hardware-in-the-Loop (HIL) simulation framework where a CARLA simulator running on a PC streams camera frames to an NVIDIA Jetson Nano edge device for real-time AI inference.

The objective is to replicate a real autonomous vehicle architecture, where sensor data is processed on an embedded edge system under realistic network conditions.

Key Features:

Simulation-based testing using CARLA
Distributed edge-AI inference on Jetson Nano
Real-time camera frame streaming
TCP vs UDP communication benchmarking
Latency (RTT) measurement and analysis
TensorRT-optimized object detection


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
        │
        ▼
Detection Results / Feedback

Communication Pipeline:

Frame Transmission (PC Side)
CARLA generates RGB camera frames
Frames are converted to NumPy arrays
JPEG compression is applied
Data is transmitted via TCP/UDP socket
Packet Structure
| Frame ID | Timestamp | Image Size | JPEG Image Data |

Frame Processing (Jetson Nano)

Receive frame via TCP/UDP
Decode JPEG image
Convert to CUDA memory
Run TensorRT inference
Output detection results

AI Model (Jetson Nano):
Feature	Description
Model	SSD-Mobilenet-V2
Dataset	MS COCO
Classes	91 categories
Runtime	TensorRT
Precision	FP16
Input Size	300 × 300

Technologies Used:
Component	Technology
Simulator	CARLA
Edge Hardware	NVIDIA Jetson Nano
AI Runtime	TensorRT
Model	SSD-Mobilenet-V2
Communication	TCP / UDP Sockets
Programming	Python
Image Processing	OpenCV

Key Insights
TCP achieves slightly lower steady-state latency (~60 ms)
UDP provides more consistent performance with fewer spikes
TCP suffers from high initial latency due to connection overhead
Jetson inference time (~37–45 ms) contributes significantly to total latency

Author

Faysal Ahammed Tonmoy
BSc Electronic Engineering
Hochschule Hamm-Lippstadt

Embedded Systems | Edge AI | Autonomous Systems
