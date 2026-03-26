import socket
import struct
import time
import csv
import cv2
import numpy as np
import jetson_inference
import jetson_utils

HOST = "0.0.0.0"
PORT = 5002

MODEL_NAME = "ssd-mobilenet-v2"
THRESHOLD = 0.5

CSV_FILE = "udp_yolo_results.csv"
MAX_UDP_PACKET = 65535
FRAME_TIMEOUT = 2.0
PRINT_EVERY = 1

# packet header format:
# [4 bytes frame_id][2 bytes chunk_id][2 bytes total_chunks][8 bytes send_ts]
HEADER_FORMAT = "!IHHd"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


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


def cleanup_old_frames(frames):
    now = time.time()
    stale_ids = []
    for frame_id, info in frames.items():
        if now - info["last_update"] > FRAME_TIMEOUT:
            stale_ids.append(frame_id)

    for frame_id in stale_ids:
        del frames[frame_id]


def main():
    print("[INFO] Loading detection network...")
    net = jetson_inference.detectNet(MODEL_NAME, threshold=THRESHOLD)
    print("[INFO] Model loaded.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    print(f"[INFO] UDP receiver listening on {HOST}:{PORT}")

    frames = {}

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

        processed_count = 0

        try:
            while True:
                packet, addr = sock.recvfrom(MAX_UDP_PACKET)
                recv_ts = time.time()

                if len(packet) < HEADER_SIZE:
                    print("[WARN] Packet too small, skipping.")
                    continue

                frame_id, chunk_id, total_chunks, send_ts = struct.unpack(
                    HEADER_FORMAT, packet[:HEADER_SIZE]
                )
                chunk_data = packet[HEADER_SIZE:]

                if frame_id not in frames:
                    frames[frame_id] = {
                        "send_ts": send_ts,
                        "total_chunks": total_chunks,
                        "chunks": {},
                        "first_recv_ts": recv_ts,
                        "last_update": recv_ts,
                    }

                frames[frame_id]["chunks"][chunk_id] = chunk_data
                frames[frame_id]["last_update"] = recv_ts

                if len(frames[frame_id]["chunks"]) == frames[frame_id]["total_chunks"]:
                    frame_info = frames[frame_id]

                    jpeg_bytes = b"".join(
                        frame_info["chunks"][i] for i in range(frame_info["total_chunks"])
                    )

                    result = process_frame(net, jpeg_bytes)
                    if result is None:
                        print(f"[WARN] Frame {frame_id} decode failed.")
                        del frames[frame_id]
                        continue

                    network_ms = (frame_info["first_recv_ts"] - frame_info["send_ts"]) * 1000.0
                    total_after_recv_ms = result["decode_ms"] + result["infer_ms"]
                    end_to_end_ms = (result["infer_end_ts"] - frame_info["send_ts"]) * 1000.0

                    writer.writerow([
                        frame_id,
                        frame_info["send_ts"],
                        frame_info["first_recv_ts"],
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

                    if processed_count % PRINT_EVERY == 0:
                        print(
                            f"[UDP][Frame {frame_id}] "
                            f"net={network_ms:.2f} ms | "
                            f"decode={result['decode_ms']:.2f} ms | "
                            f"infer={result['infer_ms']:.2f} ms | "
                            f"end2end={end_to_end_ms:.2f} ms | "
                            f"detections={result['num_detections']}"
                        )

                    processed_count += 1
                    del frames[frame_id]

                cleanup_old_frames(frames)

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")
        finally:
            sock.close()
            print(f"[INFO] Results saved to {CSV_FILE}")


if __name__ == "__main__":
    main()
