import socket
import struct
import cv2
import numpy as np
import jetson_inference
import jetson_utils

HOST = "0.0.0.0"
PORT = 5001

net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.3)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

print("[JETSON] Waiting for UDP frames")

while True:

    packet, addr = sock.recvfrom(65536)

    header = packet[:16]
    payload = packet[16:]

    frame_id, timestamp, size = struct.unpack("!IdI", header)

    jpg_array = np.frombuffer(payload, dtype=np.uint8)

    frame_bgr = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        continue

    frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)

    cuda_img = jetson_utils.cudaFromNumpy(frame_rgba)

    detections = net.Detect(cuda_img)

    print(f"[JETSON] frame={frame_id} detections={len(detections)}")

    for d in detections:

        class_name = net.GetClassDesc(d.ClassID)

        print(f"{class_name} conf={d.Confidence:.2f}")
