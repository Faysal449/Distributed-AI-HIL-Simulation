import socket
import struct
import json
import cv2
import numpy as np
import jetson_inference
import jetson_utils

HOST = "0.0.0.0"
FRAME_PORT = 5001
FEEDBACK_PORT = 5002

net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.30)

def recv_exact(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, FRAME_PORT))
server.listen(1)

print(f"[JETSON] Waiting for frame stream on {HOST}:{FRAME_PORT}")
conn, addr = server.accept()
pc_ip = addr[0]
print(f"[JETSON] Frame connection from {pc_ip}")

feedback_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
feedback_sock.connect((pc_ip, FEEDBACK_PORT))
print(f"[JETSON] Feedback connected to {pc_ip}:{FEEDBACK_PORT}")

try:
    while True:
        header = recv_exact(conn, 16)   # uint32 frame_id, double timestamp, uint32 size
        if header is None:
            print("[JETSON] Frame connection closed")
            break

        frame_id, timestamp, payload_size = struct.unpack("!IdI", header)
        payload = recv_exact(conn, payload_size)
        if payload is None:
            print("[JETSON] Payload receive failed")
            break

        jpg_array = np.frombuffer(payload, dtype=np.uint8)
        frame_bgr = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue

        frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        cuda_img = jetson_utils.cudaFromNumpy(frame_rgba)

        detections = net.Detect(cuda_img)

        result = {
            "frame": int(frame_id),
            "timestamp": float(timestamp),
            "count": len(detections),
            "detections": []
        }

        print(f"\n[JETSON] frame={frame_id} detections={len(detections)}")

        for d in detections:
            class_name = net.GetClassDesc(d.ClassID)
            item = {
                "class": class_name,
                "confidence": round(float(d.Confidence), 3),
                "bbox": [
                    round(float(d.Left), 1),
                    round(float(d.Top), 1),
                    round(float(d.Right), 1),
                    round(float(d.Bottom), 1),
                ],
            }
            result["detections"].append(item)
            print(f"  {class_name} conf={d.Confidence:.2f} box={item['bbox']}")

        msg = (json.dumps(result) + "\n").encode("utf-8")
        feedback_sock.sendall(msg)

except KeyboardInterrupt:
    print("\n[JETSON] Stopping...")
finally:
    conn.close()
    feedback_sock.close()
    server.close()
    print("[JETSON] Clean exit")
