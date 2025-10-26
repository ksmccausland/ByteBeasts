import cv2
import numpy as np
import math
import sys

def angle_deg(p0, p1, p2):
    v1 = p0 - p1
    v2 = p2 - p1
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cosang = np.dot(v1, v2) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def is_square(poly, angle_tol=12, aspect_tol=0.15, min_area=400):
    if len(poly) != 4:
        return False
    pts = poly.reshape(-1, 2).astype(np.float32)
    if not cv2.isContourConvex(poly):
        return False
    area = cv2.contourArea(poly)
    if area < min_area:
        return False

    angles = []
    for i in range(4):
        p0, p1, p2 = pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4]
        ang = angle_deg(p0, p1, p2)
        angles.append(ang)
    if not all(90 - angle_tol <= a <= 90 + angle_tol for a in angles):
        return False

    rect = cv2.minAreaRect(pts)
    (w, h) = rect[1]
    if w == 0 or h == 0:
        return False
    ratio = min(w, h) / max(w, h)
    if ratio < (1 - aspect_tol):
        return False
    return True


def detect_squares(bgr,
                   use_canny=True,
                   canny1=60, canny2=180,
                   adaptive=True,
                   blur_ksize=5):
    stage_imgs = []

    # Step 1: Grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    stage_imgs.append(("Grayscale", gray))

    # Step 2: Blur
    if blur_ksize and blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    stage_imgs.append(("Gaussian Blur", gray))

    masks = []

    # Step 3: Adaptive threshold
    if adaptive:
        thr = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 3)
        if np.mean(thr) < 127:
            thr = 255 - thr
        stage_imgs.append(("Adaptive Threshold", thr))
        masks.append(thr)

    # Step 4: Canny
    if use_canny:
        edges = cv2.Canny(gray, canny1, canny2)
        stage_imgs.append(("Canny Edges", edges))
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        stage_imgs.append(("Dilated Edges", edges))
        masks.append(edges)

    # Step 5: Combine masks
    if not masks:
        masks = [gray]
    bin_img = masks[0].copy()
    for m in masks[1:]:
        bin_img = cv2.bitwise_or(bin_img, m)
    stage_imgs.append(("Combined Mask", bin_img))

    # Step 6: Morph cleanup
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    stage_imgs.append(("Morphological Cleanup", bin_img))

    # Step 7: Find squares
    cnts_info = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

    squares = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        if peri < 30:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if is_square(approx):
            squares.append(approx)

    out = bgr.copy()
    cv2.polylines(out, squares, isClosed=True, color=(0, 255, 0), thickness=2)
    stage_imgs.append(("Final Squares", out))

    print(f"Found {len(squares)} square(s).")
    show_side_by_side(stage_imgs)
    return squares


def show_side_by_side(stages):
    """Display all stages side-by-side in one combined image."""
    # Convert grayscale images to BGR for stacking
    visuals = []
    for title, img in stages:
        if len(img.shape) == 2:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img.copy()
        visuals.append((title, img_color))

    # Resize to same height
    h = 250  # display height
    resized = []
    for title, img in visuals:
        ratio = h / img.shape[0]
        w = int(img.shape[1] * ratio)
        resized.append((title, cv2.resize(img, (w, h))))

    # Stack horizontally (if too many, stack in two rows)
    num = len(resized)
    halfway = (num + 1) // 2
    row1 = [r[1] for r in resized[:halfway]]
    row2 = [r[1] for r in resized[halfway:]] if num > halfway else []

    if row2:
        top = np.hstack(row1)
        bottom = np.hstack(row2)
        combined = np.vstack([top, bottom])
    else:
        combined = np.hstack(row1)

    # Add titles on top of each stage
    x_offset = 0
    for title, img in resized[:halfway]:
        cv2.putText(combined, title, (x_offset + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        x_offset += img.shape[1]
    if row2:
        x_offset = 0
        for title, img in resized[halfway:]:
            cv2.putText(combined, title, (x_offset + 5, h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            x_offset += img.shape[1]

    cv2.imshow("Processing Steps (press any key to close)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # You can hard-code your image file path here
    img_path = "images/test.jpg"   # or "images/test.jpg" if it’s in a folder

    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"⚠️ Could not load image: {img_path}")

    detect_squares(img)
