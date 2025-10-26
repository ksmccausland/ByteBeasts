# detect_quads_side_by_side.py
import cv2
import numpy as np
import math
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# ---------------------- Geometry helpers ----------------------
def angle_deg(p0, p1, p2):
    """Angle at p1 (degrees) formed by p0-p1-p2."""
    v1 = p0 - p1
    v2 = p2 - p1
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cosang = float(np.dot(v1, v2) / denom)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

def is_right_angle_quad(pts, angle_tol=12):
    """All 4 interior angles ~ 90° within tolerance."""
    angles = []
    for i in range(4):
        p0, p1, p2 = pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4]
        angles.append(angle_deg(p0, p1, p2))
    return all(90 - angle_tol <= a <= 90 + angle_tol for a in angles)

def classify_square_vs_rectangle(pts, square_aspect_tol=0.15):
    """
    Use minAreaRect for rotation-invariant sides.
    Return "square" if sides ~ equal; else "rectangle".
    """
    rect = cv2.minAreaRect(pts.astype(np.float32))
    w, h = rect[1]
    if w == 0 or h == 0:
        return None
    ratio = min(w, h) / max(w, h)  # 1.0 means perfect square
    return "square" if ratio >= (1.0 - square_aspect_tol) else "rectangle"

# ---------------------- Detection ----------------------
def detect_quads(
    bgr,
    use_canny=True,
    canny1=60, canny2=180,
    adaptive=True,
    blur_ksize=5,
    min_perimeter=30,
    min_area=400,
    angle_tol=12,
    square_aspect_tol=0.15
):
    stages = []

    # 1) Gray
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    stages.append(("Grayscale", gray))

    # 2) Blur
    if blur_ksize and blur_ksize > 1:
        gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        gray_blur = gray.copy()
    stages.append(("Gaussian Blur", gray_blur))

    masks = []

    # 3) Adaptive threshold (robust to lighting)
    if adaptive:
        thr = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 3
        )
        # Heuristic invert so shapes are white-ish
        if np.mean(thr) < 127:
            thr = 255 - thr
        stages.append(("Adaptive Threshold", thr))
        masks.append(thr)

    # 4) Canny edges
    if use_canny:
        edges = cv2.Canny(gray_blur, canny1, canny2)
        stages.append(("Canny Edges", edges))
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        stages.append(("Dilated Edges", edges))
        masks.append(edges)

    # 5) Combine masks
    if not masks:
        masks = [gray_blur]
    bin_img = masks[0].copy()
    for m in masks[1:]:
        bin_img = cv2.bitwise_or(bin_img, m)
    stages.append(("Combined Mask", bin_img))

    # 6) Morph cleanup
    kernel = np.ones((3, 3), np.uint8)
    bin_closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    stages.append(("Morphological Cleanup", bin_closed))

    # 7) Contours -> quads
    cnts_info = cv2.findContours(bin_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

    squares = []
    rectangles = []

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

    # 8) Build final overlay
    overlay = bgr.copy()
    if rectangles:
        cv2.polylines(overlay, rectangles, isClosed=True, color=(255, 0, 0), thickness=2)  # blue
    if squares:
        cv2.polylines(overlay, squares, isClosed=True, color=(0, 255, 0), thickness=2)    # green
    stages.append(("Final (green=squares, blue=rectangles)", overlay))

    return squares, rectangles, stages

# ---------------------- Visualization ----------------------
def show_side_by_side(stages, window_title="Processing Steps (press any key to close)", max_row_height=260):
    """
    Stacks all (title, image) stages into 1 or 2 rows for quick inspection.
    """
    visuals = []
    for title, img in stages:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        visuals.append((title, img))

    # Resize to uniform height
    resized = []
    h = max_row_height
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
        combined = None

    if row2_imgs:
        bottom = np.hstack(row2_imgs)
        combined = np.vstack([combined, bottom]) if combined is not None else bottom

    # Add stage titles
    # (Compute x offsets by summing widths)
    def annotate_titles(start_y, items):
        x = 5
        for title, im in items:
            cv2.putText(combined, title, (x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            x += im.shape[1]

    if combined is None:
        return None

    annotate_titles(22, resized[:half])
    if row2_imgs:
        annotate_titles(h + 22, resized[half:])

    cv2.imshow(window_title, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return combined

# ---------------------- Main (IDE-friendly) ----------------------
if __name__ == "__main__":
    # File picker for IDE use
    root = tk.Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    if not img_path:
        raise SystemExit("No file selected.")

    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"Could not load image: {img_path}")

    squares, rectangles, stages = detect_quads(
        img,
        use_canny=True,
        canny1=60, canny2=180,
        adaptive=True,
        blur_ksize=5,
        min_perimeter=30,
        min_area=400,
        angle_tol=12,
        square_aspect_tol=0.15
    )

    print(f"Squares: {len(squares)} | Rectangles: {len(rectangles)}")

    composite = show_side_by_side(stages)

    # Optional: save outputs next to the image
    if composite is not None:
        out_dir = Path(img_path).parent
        cv2.imwrite(str(out_dir / "pipeline_debug.jpg"), composite)
        # Last stage is overlay; save it separately too
        cv2.imwrite(str(out_dir / "detections_overlay.jpg"), stages[-1][1])
        print(f"Saved: {out_dir / 'pipeline_debug.jpg'}")
        print(f"Saved: {out_dir / 'detections_overlay.jpg'}")
