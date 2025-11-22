import argparse
import time
import sys
import warnings

# This hides a noisy pygame warning about "pkg_resources"
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

# winsound is only on Windows. We use it for a simple emergency beep.
try:
    import winsound
except ImportError:
    winsound = None

import numpy as np
import cv2
import pygame
from ultralytics import YOLO

# ---------- BASIC SETTINGS ----------
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 600

CAM_W = 520      # webcam preview width in pygame window
CAM_H = 390      # webcam preview height in pygame window
CAM_X = 20       # position to draw camera preview
CAM_Y = 80

SIGNAL_X = 700   # position to draw signal
SIGNAL_Y = 80

MIN_GREEN = 8    # minimum green time
MAX_GREEN = 40   # maximum green time
TIME_PER_VEHICLE = 1.2  # extra seconds per vehicle

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


def play_emergency_beep():
    """Simple emergency beep. If winsound is not available, do nothing."""
    if winsound is None:
        print("Emergency mode ON (no sound on this system).")
        return
    winsound.Beep(1200, 300)


def count_vehicles(result, names, conf_threshold):
    """
    Count cars, bikes and trucks from YOLO result.
    result: one frame result from model(frame)[0]
    names: model.names (id -> class name)
    """
    counts = {"cars": 0, "bikes": 0, "trucks": 0}

    if result is None or result.boxes is None or len(result.boxes) == 0:
        return counts

    boxes = result.boxes
    cls_ids = boxes.cls.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    for i in range(len(cls_ids)):
        cls_id = int(cls_ids[i])
        conf = float(confs[i])

        if conf < conf_threshold:
            continue

        name = names[cls_id]

        if name == "car":
            counts["cars"] += 1
        elif name in ("motorcycle", "bicycle"):
            counts["bikes"] += 1
        elif name in ("truck", "bus"):
            counts["trucks"] += 1

    return counts


def calc_green_time(counts):
    """Return green time in seconds based on vehicle counts."""
    total = counts["cars"] + counts["bikes"] + counts["trucks"]
    green = int(total * TIME_PER_VEHICLE)

    if green < MIN_GREEN:
        green = MIN_GREEN
    if green > MAX_GREEN:
        green = MAX_GREEN

    return green


def draw_signal(screen, x, y, state):
    """Draw a very simple traffic signal."""
    GREY = (60, 60, 60)
    RED = (220, 0, 0)
    YELLOW = (220, 220, 0)
    GREEN = (0, 220, 0)
    DARK_RED = (80, 0, 0)
    DARK_YELLOW = (80, 80, 0)
    DARK_GREEN = (0, 80, 0)

    pygame.draw.rect(screen, GREY, (x, y, 120, 300), border_radius=10)

    cx = x + 60
    r1 = (cx, y + 60)
    r2 = (cx, y + 150)
    r3 = (cx, y + 240)

    # Only one of these will be bright
    pygame.draw.circle(screen, RED if state == "RED" else DARK_RED, r1, 35)
    pygame.draw.circle(screen, YELLOW if state == "YELLOW" else DARK_YELLOW, r2, 35)
    pygame.draw.circle(screen, GREEN if state == "GREEN" else DARK_GREEN, r3, 35)


def frame_to_surface(frame):
    """Convert OpenCV frame (BGR) to Pygame surface (RGB) with fixed size."""
    frame_resized = cv2.resize(frame, (CAM_W, CAM_H))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    surf = pygame.image.frombuffer(frame_rgb.tobytes(), (CAM_W, CAM_H), "RGB")
    return surf


def main(args):
    # ---------- INIT PYGAME ----------
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Smart Traffic Controller - Beginner Version")

    font = pygame.font.SysFont("Arial", 20)
    small_font = pygame.font.SysFont("Arial", 16)
    clock = pygame.time.Clock()

    # ---------- LOAD YOLO MODEL ----------
    print("Loading YOLO model:", args.model)
    model = YOLO(args.model)   # will download if needed

    # ---------- OPEN CAMERA OR VIDEO ----------
    if args.source is None:
        # webcam
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print("Could not open camera or video file.")
        return

    # ---------- TRAFFIC LOGIC SETUP ----------
    directions = ["North", "East", "South", "West"]
    current_dir = 0

    emergency_mode = False
    green_time_left = 10
    last_second = time.time()

    vehicles = {"cars": 0, "bikes": 0, "trucks": 0}

    running = True

    # ---------- MAIN LOOP ----------
    while running:
        # --- HANDLE EVENTS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_e:
                    # toggle emergency mode
                    emergency_mode = not emergency_mode
                    if emergency_mode:
                        play_emergency_beep()

        # --- READ FRAME ---
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        # --- YOLO DETECTION ---
        results = model(frame)[0]          # one frame result
        vehicles = count_vehicles(results, model.names, args.conf)

        # --- UPDATE GREEN TIMER ---
        # emergency mode: fixed 6 seconds
        if emergency_mode:
            target_green = 6
        else:
            target_green = calc_green_time(vehicles)

        # reduce timer once per real second
        now = time.time()
        if now - last_second >= 1.0:
            last_second = now
            green_time_left -= 1

            if green_time_left <= 0:
                # move to next direction
                current_dir = (current_dir + 1) % len(directions)
                green_time_left = target_green

        # --- DRAW EVERYTHING ---
        screen.fill((30, 30, 30))

        # camera preview
        cam_surf = frame_to_surface(frame)
        screen.blit(cam_surf, (CAM_X, CAM_Y))

        # draw YOLO boxes (only on preview area)
        if results.boxes is not None and len(results.boxes) > 0:
            h_orig, w_orig = frame.shape[:2]
            x_scale = CAM_W / w_orig
            y_scale = CAM_H / h_orig

            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()

            for i, box in enumerate(xyxy):
                conf = float(confs[i])
                cls_id = int(clss[i])
                name = model.names[cls_id]

                if conf < args.conf:
                    continue
                if name not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = box
                rx1 = int(x1 * x_scale) + CAM_X
                ry1 = int(y1 * y_scale) + CAM_Y
                rx2 = int(x2 * x_scale) + CAM_X
                ry2 = int(y2 * y_scale) + CAM_Y

                pygame.draw.rect(
                    screen,
                    (0, 255, 0),
                    (rx1, ry1, rx2 - rx1, ry2 - ry1),
                    2
                )

                label = f"{name} {conf:.2f}"
                text_surface = small_font.render(label, True, (255, 255, 255))
                screen.blit(text_surface, (rx1, ry1 - 18))

        # traffic signal state: simple (green if timer > 0)
        signal_state = "GREEN" if green_time_left > 0 else "RED"
        draw_signal(screen, SIGNAL_X, SIGNAL_Y, signal_state)

        # text info
        text_dir = font.render(f"Current Signal: {directions[current_dir]}", True, (255, 255, 255))
        text_counts = font.render(
            f"Cars: {vehicles['cars']}  Bikes: {vehicles['bikes']}  Trucks/Buses: {vehicles['trucks']}",
            True, (255, 255, 255)
        )
        text_timer = font.render(f"Green Time Left: {max(green_time_left, 0)} sec", True, (255, 255, 255))

        if emergency_mode:
            mode_msg = "EMERGENCY MODE (Press E to switch off)"
        else:
            mode_msg = "Normal Mode (Press E for emergency)"

        text_mode = font.render(mode_msg, True, (255, 200, 0))
        text_help = small_font.render("Press Q to quit | E to toggle emergency", True, (200, 200, 200))

        screen.blit(text_dir, (20, 20))
        screen.blit(text_counts, (20, 490))
        screen.blit(text_timer, (20, 520))
        screen.blit(text_mode, (20, 550))
        screen.blit(text_help, (540, 20))

        pygame.display.flip()
        clock.tick(30)   # limit to ~30 FPS

    # ---------- CLEANUP ----------
    cap.release()
    pygame.quit()


if __name__ == "_main_":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        "-s",
        help="Video file path (optional). If not given, webcam is used.",
        default=None
    )
    parser.add_argument(
        "--model",
        "-m",
        help="YOLO model file (example: yolov8n.pt)",
        default="yolov8n.pt"
    )
    parser.add_argument(
        "--conf",
        help="Confidence threshold (0 to 1).",
        type=float,
        default=0.35,
    )

    args = parser.parse_args()
    main(args)
