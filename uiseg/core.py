import cv2
import numpy as np
from pydantic import BaseModel
from PIL import Image, ImageDraw


class UISegConfig(BaseModel):
    """
    Configuration for UISeg.
    
    Attributes:
        min_length_ratio (float): Ratio of minimum width/height to image size
        max_area_ratio (float): Maximum allowed area ratio of a region relative to the image
        merge_max_gap_ratio (float): Ratio of maximum horizontal gap to image width for merging
        merge_min_overlap_ratio (float): Minimum vertical overlap ratio for merging adjacent regions
        adaptive_block_size (int): Block size for adaptive thresholding (must be odd)
        adaptive_C (int): Constant subtracted from mean in adaptive thresholding
        morph_kernel_size (int): Kernel size for morphological operations
        morph_iterations (int): Number of iterations for morphological operations
    """

    min_length_ratio: float = 0.005
    max_area_ratio: float = 0.4
    merge_max_gap_ratio: float = 0.005
    merge_min_overlap_ratio: float = 0.5

    # internal
    adaptive_block_size: int = 11
    adaptive_C: int = 2
    morph_kernel_size: int = 3
    morph_iterations: int = 1


class UISeg:
    def __init__(self, config: UISegConfig = UISegConfig()):
        self.config = config

    def process_image(self, image: np.ndarray, show: bool = False):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply adaptive thresholding to get binary image
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.config.adaptive_block_size, self.config.adaptive_C
        )

        # Morphological dilation to connect regions
        kernel = np.ones((self.config.morph_kernel_size, self.config.morph_kernel_size), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=self.config.morph_iterations)

        # Connected components analysis to find candidate regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        raw_regions = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            raw_regions.append((x, y, w, h, area))

        # Filter regions by size and area ratio
        img_area = image.shape[0] * image.shape[1]
        img_h, img_w = image.shape[:2]
        min_length = int(min(img_w, img_h) * self.config.min_length_ratio)
        merge_max_gap = int(img_w * self.config.merge_max_gap_ratio)
        filtered = []
        for x, y, w, h, _ in raw_regions:
            area = w * h
            if w < min_length or h < min_length:
                continue
            if area / img_area > self.config.max_area_ratio:
                continue
            filtered.append((x, y, w, h, area))

        # Remove regions that are inside other regions
        def is_inside(r1, r2):
            x1, y1, w1, h1, _ = r1
            x2, y2, w2, h2, _ = r2
            return x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2

        keep = []
        for i, r1 in enumerate(filtered):
            inside = False
            for j, r2 in enumerate(filtered):
                if i != j and is_inside(r1, r2):
                    inside = True
                    break
            if not inside:
                keep.append(r1)

        # Prepare region list for merging
        regions = [(x, y, w, h) for x, y, w, h, area in keep]

        # Define merging criteria for horizontally adjacent regions
        def can_merge(r1, r2, max_gap=None, min_overlap_ratio=None):
            x1, y1, w1, h1 = r1
            x2, y2, w2, h2 = r2
            if not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1):
                return True
            if max_gap is None:
                max_gap = merge_max_gap
            if min_overlap_ratio is None:
                min_overlap_ratio = self.config.merge_min_overlap_ratio
            y_top = max(y1, y2)
            y_bot = min(y1 + h1, y2 + h2)
            overlap = max(0, y_bot - y_top)
            min_h = min(h1, h2)
            if min_h == 0:
                return False
            overlap_ratio = overlap / min_h
            if x2 > x1:
                gap = x2 - (x1 + w1)
                x_gap_start = x1 + w1
                x_gap_end = x2
            else:
                gap = x1 - (x2 + w2)
                x_gap_start = x2 + w2
                x_gap_end = x1
            if not (0 <= gap <= max_gap and overlap_ratio >= min_overlap_ratio):
                return False

            # Check if the gap between regions is mostly white (background)
            if x_gap_end > x_gap_start and overlap > 0:
                y_start = y_top
                y_end = y_bot
                gap_roi = binary[y_start:y_end, x_gap_start:x_gap_end]
                if gap_roi.size > 0:
                    white_ratio = np.mean(gap_roi > 0)
                    if white_ratio > 0.6:
                        return False

            # Check edge strength to avoid merging strong separated regions
            def edge_strength(region):
                x, y, w, h = region
                roi = binary[y:y + h, x:x + w]
                left_edge = roi[:, :2]
                right_edge = roi[:, -2:]
                left_ratio = np.mean(left_edge > 0)
                right_ratio = np.mean(right_edge > 0)
                return left_ratio, right_ratio

            r1_right, _ = edge_strength(r1)
            _, r2_left = edge_strength(r2)
            if r1_right > 0.6 and r2_left > 0.6:
                return False
            return True

        # Sort regions by x coordinate for merging
        regions.sort(key=lambda r: r[0])
        merged = []
        used = [False] * len(regions)
        for i in range(len(regions)):
            if used[i]:
                continue
            x, y, w, h = regions[i]
            cur = [x, y, w, h]
            for j in range(i + 1, len(regions)):
                if used[j]:
                    continue
                if can_merge(cur, regions[j]):
                    nx, ny, nw, nh = regions[j]
                    x1 = min(cur[0], nx)
                    y1 = min(cur[1], ny)
                    x2 = max(cur[0] + cur[2], nx + nw)
                    y2 = max(cur[1] + cur[3], ny + nh)
                    cur = [x1, y1, x2 - x1, y2 - y1]
                    used[j] = True
            merged.append(tuple(cur))
            used[i] = True
        regions = merged

        # Optionally show the detected regions on the image
        if show:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            for idx, (x, y, w, h) in enumerate(regions):
                draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)
                draw.text((x, y), str(idx + 1), fill=(255, 0, 0))
            pil_img.show()

        return regions

    def process_image_file(self, image_path: str, *args, **kwargs):
        image = cv2.imread(image_path)
        return self.process_image(image, *args, **kwargs)
