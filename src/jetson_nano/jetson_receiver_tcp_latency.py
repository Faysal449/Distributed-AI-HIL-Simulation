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

HEADER_FORMAT = "!IdI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.3)


def recv_exact(sock, size):
    data = b""
    while len(data) < size:
        try:
            packet = sock.recv(size - len(data))
        except ConnectionResetError:
            return None
        except Exception as e:
            print(f"[JETSON-TCP] recv error: {e}")
            return None

        if not packet:
            return None
        data += packet
    return data


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, FRAME_PORT))
    server.listen(1)

    print(f"[JETSON-TCP] Waiting for frame stream on {HOST}:{FRAME_PORT}")
    conn, addr = server.accept()
    pc_ip = addr[0]
    print(f"[JETSON-TCP] Frame connection from {pc_ip}")

    feedback_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    feedback_sock.connect((pc_ip, FEEDBACK_PORT))
    print(f"[JETSON-TCP] Feedback connected to {pc_ip}:{FEEDBACK_PORT}")

    recv_count = 0

    try:
        while True:
            header = recv_exact(conn, HEADER_SIZE)
            if header is None:
                print("[JETSON-TCP] Connection closed by PC")
                break

            frame_id, pc_send_time, payload_size = struct.unpack(HEADER_FORMAT, header)

            payload = recv_exact(conn, payload_size)
            if payload is None:
                print("[JETSON-TCP] Payload receive failed / connection closed")
                break

            jetson_recv_time = time.time()

            jpg_array = np.frombuffer(payload, dtype=np.uint8)
            frame_bgr = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                print(f"[JETSON-TCP] JPEG decode failed for frame={frame_id}")
                continue

            frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
            cuda_img = jetson_utils.cudaFromNumpy(frame_rgba.copy())

            t0 = time.time()
            detections = net.Detect(cuda_img)
            t1 = time.time()

            inference_ms = (t1 - t0) * 1000.0
            recv_count += 1

            result = {
                "protocol": "tcp",
                "frame": int(frame_id),
                "pc_send_time": float(pc_send_time),
                "jetson_recv_time": float(jetson_recv_time),
                "jetson_done_time": float(t1),
                "inference_ms": round(inference_ms, 3),
                "count": len(detections),
                "recv_count": recv_count,
                "jpeg_bytes": payload_size
            }

            try:
                feedback_sock.sendall((json.dumps(result) + "\n").encode("utf-8"))
            except BrokenPipeError:
                print("[JETSON-TCP] Feedback socket closed by PC")
                break
            except Exception as e:
                print(f"[JETSON-TCP] Feedback send error: {e}")
                break

            print(
                f"[JETSON-TCP] frame={frame_id} "
                f"recv_count={recv_count} "
                f"bytes={payload_size} "
                f"det={len(detections)} "
                f"inf={inference_ms:.2f} ms"
            )

    except KeyboardInterrupt:
        print("\n[JETSON-TCP] Stopping...")

    finally:
        try:
            conn.close()
        except Exception:
            pass

        try:
            feedback_sock.close()
        except Exception:
            pass

        try:
            server.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
