import socket
import struct
import json
import time
import cv2
import numpy as np
import jetson_inference
import jetson_utils

HOST = "0.0.0.0"
FRAME_PORT = 5001
FEEDBACK_PORT = 5002

net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.3)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, FRAME_PORT))

feedback_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

recv_count = 0
print("[JETSON-UDP] Waiting for UDP frames")

while True:
    packet, addr = sock.recvfrom(65535)
    jetson_recv_time = time.time()

    if len(packet) < 16:
        continue

    header = packet[:16]
    payload = packet[16:]

    frame_id, pc_send_time, payload_size = struct.unpack("!IdI", header)

    if len(payload) != payload_size:
        continue

    jpg_array = np.frombuffer(payload, dtype=np.uint8)
    frame_bgr = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        continue

    frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    cuda_img = jetson_utils.cudaFromNumpy(frame_rgba)

    t0 = time.time()
    detections = net.Detect(cuda_img)
    t1 = time.time()

    recv_count += 1

    result = {
        "protocol": "udp",
        "frame": int(frame_id),
        "pc_send_time": float(pc_send_time),
        "jetson_recv_time": float(jetson_recv_time),
        "jetson_done_time": float(t1),
        "inference_ms": round((t1 - t0) * 1000.0, 3),
        "count": len(detections),
        "recv_count": recv_count,
    }

    feedback_sock.sendto((json.dumps(result) + "\n").encode("utf-8"), (addr[0], FEEDBACK_PORT))
    print(f"[JETSON-UDP] frame={frame_id} det={len(detections)}")
