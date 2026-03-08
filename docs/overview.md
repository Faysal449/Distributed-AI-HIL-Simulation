# Project Overview

This project implements a Distributed Hardware-in-the-Loop (HIL) simulation
framework for autonomous driving using CARLA and Jetson Nano.

The simulator runs on a Windows PC while perception and processing are
executed on an edge device (Jetson Nano). Both systems communicate over
a TCP network.

The goal is to simulate a realistic autonomous vehicle architecture where
sensor data is processed on embedded hardware while interacting with a
high-fidelity simulator in real time.
