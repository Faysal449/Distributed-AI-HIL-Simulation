import socket
import struct
import time
import csv
import cv2
import numpy as np
import jetson_inference
import jetson_utils

HOST = "0.0.0.0"
PORT = 5001

MODEL_NAME = "ssd-mobilenet-v2"   # change later if needed
THRESHOLD = 0.5

CSV_FILE = "tcp_yolo_results.csv"
PRINT_EVERY = 1


def recv_exact(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data


def process_frame(net, jpeg_bytes):
    decode_start = time.time()

    np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    decode_end = time.time()

    if frame_bgr is None:
        return None

    frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    cuda_img = jetson_utils.cudaFromNumpy(frame_rgba)

    infer_start = time.time()
    detections = net.Detect(cuda_img, overlay="none")
    infer_end = time.time()

    return {
        "width": frame_bgr.shape[1],
        "height": frame_bgr.shape[0],
        "decode_ms": (decode_end - decode_start) * 1000.0,
        "infer_ms": (infer_end - infer_start) * 1000.0,
        "num_detections": len(detections),
        "infer_end_ts": infer_end,
    }


def main():
    print("[INFO] Loading detection network...")
    net = jetson_inference.detectNet(MODEL_NAME, threshold=THRESHOLD)
    print("[INFO] Model loaded.")

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_id",
            "send_ts",
            "recv_ts",
            "network_ms",
            "decode_ms",
            "infer_ms",
            "total_after_recv_ms",
            "end_to_end_ms",
            "num_detections",
            "width",
            "height"
        ])

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        print(f"[INFO] TCP receiver listening on {HOST}:{PORT}")
        conn, addr = server.accept()
        print(f"[INFO] Connected by {addr}")

        frame_id = 0

        try:
            while True:
                # expected TCP packet format:
                # [4 bytes payload_size][8 bytes send_timestamp][jpeg_bytes]
                header = recv_exact(conn, 4)
                if header is None:
                    print("[INFO] Connection closed.")
                    break

                payload_size = struct.unpack("!I", header)[0]
                payload = recv_exact(conn, payload_size)
                if payload is None:
                    print("[INFO] Incomplete payload received.")
                    break

                recv_ts = time.time()

                if len(payload) < 8:
                    print("[WARN] Payload too small, skipping.")
                    continue

                send_ts = struct.unpack("!d", payload[:8])[0]
                jpeg_bytes = payload[8:]

                result = process_frame(net, jpeg_bytes)
                if result is None:
                    print(f"[WARN] Frame {frame_id} decode failed.")
                    continue

                network_ms = (recv_ts - send_ts) * 1000.0
                total_after_recv_ms = result["decode_ms"] + result["infer_ms"]
                end_to_end_ms = (result["infer_end_ts"] - send_ts) * 1000.0

                writer.writerow([
                    frame_id,
                    send_ts,
                    recv_ts,
                    network_ms,
                    result["decode_ms"],
                    result["infer_ms"],
                    total_after_recv_ms,
                    end_to_end_ms,
                    result["num_detections"],
                    result["width"],
                    result["height"]
                ])
                f.flush()

                if frame_id % PRINT_EVERY == 0:
                    print(
                        f"[TCP][Frame {frame_id}] "
                        f"net={network_ms:.2f} ms | "
                        f"decode={result['decode_ms']:.2f} ms | "
                        f"infer={result['infer_ms']:.2f} ms | "
                        f"end2end={end_to_end_ms:.2f} ms | "
                        f"detections={result['num_detections']}"
                    )

                frame_id += 1

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")
        finally:
            conn.close()
            server.close()
            print(f"[INFO] Results saved to {CSV_FILE}")


if __name__ == "__main__":
    main()
