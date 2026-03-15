Distributed Edge-AI Hardware-in-the-Loop (HIL) Simulation using CARLA & Jetson Nano
Overview

This project implements a Distributed Hardware-in-the-Loop (HIL) simulation pipeline where a CARLA simulator running on a PC streams camera frames to an NVIDIA Jetson Nano edge device for real-time AI object detection.

The goal is to simulate a real autonomous vehicle architecture, where sensor data from a vehicle is processed by an edge AI system.

The system demonstrates:

Simulation-based testing of perception algorithms

Distributed edge-AI inference

Real-time sensor streaming

Communication between a simulator and embedded AI hardware
Real-time sensor streaming

Communication between a simulator and embedded AI hardware


System Architecture
CARLA Simulator (PC)
        │
        │ RGB Camera Sensor
        ▼
Python CARLA Client
        │
        │ JPEG Encoding
        ▼
TCP Socket Communication
        │
        │ Ethernet / WiFi Network
        ▼
Jetson Nano Edge Device
        │
        │ TensorRT Inference
        ▼
AI Object Detection
        │
        ▼
Detection Results / Feedback

Communication Process

The system uses TCP socket communication between the PC and the Jetson Nano.

Frame Transmission

CARLA generates camera frames.

The PC sender script converts the frame into a NumPy image.

The image is compressed using JPEG encoding.

The frame is transmitted via TCP socket to the Jetson Nano.

Packet structure:

| Frame ID | Timestamp | Image Size | JPEG Image Data |

Frame Processing on Jetson

The Jetson Nano performs the following steps:

Receive image frame via TCP

Decode JPEG image

Convert image to CUDA memory

Run object detection

Return detection feedback

AI Model Running on Jetson Nano

The object detection model used is:

SSD-Mobilenet-V2
Model Details
Feature	Description
Model	SSD-Mobilenet-V2
Dataset	MS COCO
Classes	91 object categories
Runtime	TensorRT
Precision	FP16
Input Size	300×300

Technologies Used
Component	Technology
Simulator	CARLA
Edge Hardware	NVIDIA Jetson Nano
AI Runtime	TensorRT
Model	SSD-Mobilenet-V2
Communication	TCP sockets
Programming Language	Python
Image Processing	OpenCV

Author

Faysal Ahammed Tonmoy

BSc, Electronic Engineering 

Embedded Systems | Edge AI | Autonomous Systems
