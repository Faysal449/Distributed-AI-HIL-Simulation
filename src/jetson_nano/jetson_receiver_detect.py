import socket
import struct
import cv2
import numpy as np
import jetson_inference
import jetson_utils

HOST = "0.0.0.0"
PORT = 5001

net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.2)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

print(f"[JETSON] Waiting on {HOST}:{PORT}")
conn, addr = server.accept()
print(f"[JETSON] Connected by {addr}")

def recv_exact(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

try:
    while True:
        header = recv_exact(conn, 16)
        if header is None:
            break

        frame_id, timestamp, payload_size = struct.unpack("!IdI", header)
        payload = recv_exact(conn, payload_size)
        if payload is None:
            break

        jpg_array = np.frombuffer(payload, dtype=np.uint8)
        frame_bgr = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue

        frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        cuda_img = jetson_utils.cudaFromNumpy(frame_rgba)

        detections = net.Detect(cuda_img)

        print(f"\n[JETSON] frame={frame_id} detections={len(detections)}")
        for d in detections:
            print(f"  {net.GetClassDesc(d.ClassID)} conf={d.Confidence:.2f}")

except KeyboardInterrupt:
    pass
finally:
    conn.close()
    server.close()
