# quad_detector_live_sliders.py
import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# ---------------------- Geometry helpers ----------------------
def angle_deg(p0, p1, p2):
    v1 = p0 - p1
    v2 = p2 - p1
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cosang = float(np.dot(v1, v2) / denom)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

def is_right_angle_quad(pts, angle_tol=12):
    angles = []
    for i in range(4):
        p0, p1, p2 = pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4]
        angles.append(angle_deg(p0, p1, p2))
    return all(90 - angle_tol <= a <= 90 + angle_tol for a in angles)

def classify_square_vs_rectangle(pts, square_aspect_tol=0.15):
    rect = cv2.minAreaRect(pts.astype(np.float32))
    w, h = rect[1]
    if w == 0 or h == 0:
        return None
    ratio = min(w, h) / max(w, h)  # 1.0 is a perfect square
    return "square" if ratio >= (1.0 - square_aspect_tol) else "rectangle"

# ---------------------- Detection core ----------------------
def detect_quads_with_params(
    bgr,
    blur_ksize=5,
    use_adaptive=True, block_size=21, C=3,
    use_canny=True, canny1=60, canny2=180,
    min_perimeter=30, min_area=400,
    angle_tol=12, square_aspect_tol=0.15
):
    stages = []

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    stages.append(("Grayscale", gray))

    if blur_ksize < 1:
        blur_ksize = 1
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0) if blur_ksize > 1 else gray
    stages.append(("Gaussian Blur", gray_blur))

    masks = []

    if use_adaptive:
        # block_size must be odd and >= 3
        if block_size < 3:
            block_size = 3
        if block_size % 2 == 0:
            block_size += 1
        thr = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            block_size, C
        )
        # Heuristic invert so shapes tend to be white foreground
        if np.mean(thr) < 127:
            thr = 255 - thr
        stages.append(("Adaptive Threshold", thr))
        masks.append(thr)

    if use_canny:
        if canny2 <= canny1:
            canny2 = canny1 + 1
        edges = cv2.Canny(gray_blur, canny1, canny2)
        stages.append(("Canny Edges", edges))
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        stages.append(("Dilated Edges", edges))
        masks.append(edges)

    if not masks:
        masks = [gray_blur]
    bin_img = masks[0].copy()
    for m in masks[1:]:
        bin_img = cv2.bitwise_or(bin_img, m)
    stages.append(("Combined Mask", bin_img))

    kernel = np.ones((3, 3), np.uint8)
    bin_closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    stages.append(("Morphological Cleanup", bin_closed))

    cnts_info = cv2.findContours(bin_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

    squares, rectangles = [], []
    for c in contours:
        peri = cv2.arcLength(c, True)
        if peri < min_perimeter:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        if area < min_area:
            continue

        pts = approx.reshape(-1, 2).astype(np.float32)
        if not is_right_angle_quad(pts, angle_tol=angle_tol):
            continue

        kind = classify_square_vs_rectangle(pts, square_aspect_tol=square_aspect_tol)
        if kind == "square":
            squares.append(approx)
        elif kind == "rectangle":
            rectangles.append(approx)

    overlay = bgr.copy()
    if rectangles:
        cv2.polylines(overlay, rectangles, isClosed=True, color=(255, 0, 0), thickness=2)  # blue
    if squares:
        cv2.polylines(overlay, squares, isClosed=True, color=(0, 255, 0), thickness=2)    # green
    stages.append(("Final (green=squares, blue=rectangles)", overlay))

    return squares, rectangles, stages

# ---------------------- Visualization ----------------------
def build_composite(stages, max_row_height=260):
    visuals = []
    for title, img in stages:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        visuals.append((title, img))

    # Resize to uniform height
    h = max_row_height
    resized = []
    for title, img in visuals:
        r = h / img.shape[0]
        w = int(img.shape[1] * r)
        resized.append((title, cv2.resize(img, (w, h))))

    # Arrange rows
    n = len(resized)
    half = (n + 1) // 2
    row1_imgs = [im for _, im in resized[:half]]
    row2_imgs = [im for _, im in resized[half:]] if n > half else []

    if row1_imgs:
        top = np.hstack(row1_imgs)
        combined = top
    else:
        return None

    if row2_imgs:
        bottom = np.hstack(row2_imgs)
        combined = np.vstack([combined, bottom])

    # Titles
    def annotate_titles(start_y, items):
        x = 5
        for title, im in items:
            cv2.putText(combined, title, (x, start_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            x += im.shape[1]

    annotate_titles(22, resized[:half])
    if row2_imgs:
        annotate_titles(h + 22, resized[half:])
    return combined

# ---------------------- Trackbar helpers ----------------------
def _noop(val): pass

def make_controls_window():
    cv2.namedWindow("Controls", cv2.WINDOW_AUTOSIZE)

    # Blur
    cv2.createTrackbar("Blur ksize (odd)", "Controls", 5, 31, _noop)  # 1..31, we force odd

    # Adaptive Threshold
    cv2.createTrackbar("Adaptive ON (0/1)", "Controls", 1, 1, _noop)
    cv2.createTrackbar("Block Size (odd)", "Controls", 21, 51, _noop)  # 3..51, we force odd
    cv2.createTrackbar("C (0..15)", "Controls", 3, 15, _noop)

    # Canny
    cv2.createTrackbar("Canny ON (0/1)", "Controls", 1, 1, _noop)
    cv2.createTrackbar("Canny Low", "Controls", 60, 255, _noop)
    cv2.createTrackbar("Canny High", "Controls", 180, 255, _noop)

    # Detection hygiene (optional but handy)
    cv2.createTrackbar("Min Area", "Controls", 400, 50000, _noop)     # pixels
    cv2.createTrackbar("Angle tol", "Controls", 12, 25, _noop)        # degrees
    cv2.createTrackbar("Square tol %", "Controls", 15, 30, _noop)     # percent (0-30)

def get_params_from_controls():
    # Read sliders
    blur = cv2.getTrackbarPos("Blur ksize (odd)", "Controls")
    if blur < 1: blur = 1
    if blur % 2 == 0: blur += 1

    use_adapt = cv2.getTrackbarPos("Adaptive ON (0/1)", "Controls") == 1
    blk = cv2.getTrackbarPos("Block Size (odd)", "Controls")
    if blk < 3: blk = 3
    if blk % 2 == 0: blk += 1
    C = cv2.getTrackbarPos("C (0..15)", "Controls")

    use_canny = cv2.getTrackbarPos("Canny ON (0/1)", "Controls") == 1
    c1 = cv2.getTrackbarPos("Canny Low", "Controls")
    c2 = cv2.getTrackbarPos("Canny High", "Controls")
    if c2 <= c1: c2 = c1 + 1

    min_area = max(0, cv2.getTrackbarPos("Min Area", "Controls"))
    angle_tol = max(1, cv2.getTrackbarPos("Angle tol", "Controls"))
    square_tol_pct = cv2.getTrackbarPos("Square tol %", "Controls")
    square_aspect_tol = max(0.01, square_tol_pct / 100.0)

    return {
        "blur_ksize": blur,
        "use_adaptive": use_adapt,
        "block_size": blk,
        "C": C,
        "use_canny": use_canny,
        "canny1": c1,
        "canny2": c2,
        "min_area": max(10, min_area),
        "angle_tol": angle_tol,
        "square_aspect_tol": square_aspect_tol
    }

# ---------------------- Main (IDE friendly + live loop) ----------------------
if __name__ == "__main__":
    # Pick file via dialog (great for IDE runs)
    root = tk.Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    if not img_path:
        raise SystemExit("No file selected.")

    bgr = cv2.imread(img_path)
    if bgr is None:
        raise SystemExit(f"Could not load image: {img_path}")

    make_controls_window()
    cv2.namedWindow("Pipeline (press s to save, q/ESC to quit)", cv2.WINDOW_NORMAL)

    saved_count = 0
    while True:
        params = get_params_from_controls()

        squares, rects, stages = detect_quads_with_params(
            bgr,
            blur_ksize=params["blur_ksize"],
            use_adaptive=params["use_adaptive"],
            block_size=params["block_size"],
            C=params["C"],
            use_canny=params["use_canny"],
            canny1=params["canny1"],
            canny2=params["canny2"],
            min_perimeter=30,
            min_area=params["min_area"],
            angle_tol=params["angle_tol"],
            square_aspect_tol=params["square_aspect_tol"]
        )

        composite = build_composite(stages, max_row_height=260)
        if composite is None:
            break

        # Optional: limit window size if huge
        max_w = 1800
        if composite.shape[1] > max_w:
            scale = max_w / composite.shape[1]
            composite_disp = cv2.resize(composite, (int(composite.shape[1]*scale), int(composite.shape[0]*scale)))
        else:
            composite_disp = composite

        cv2.imshow("Pipeline (press s to save, q/ESC to quit)", composite_disp)

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break
        if key == ord('s'):
            out_dir = Path(img_path).parent
            cv2.imwrite(str(out_dir / f"pipeline_debug_{saved_count}.jpg"), composite)
            cv2.imwrite(str(out_dir / f"detections_overlay_{saved_count}.jpg"), stages[-1][1])
            print(f"Saved: {out_dir / f'pipeline_debug_{saved_count}.jpg'}")
            print(f"Saved: {out_dir / f'detections_overlay_{saved_count}.jpg'}")
            saved_count += 1

    cv2.destroyAllWindows()
