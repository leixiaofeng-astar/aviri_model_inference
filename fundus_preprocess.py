"""
Fundus image standardized preprocessing — crop the retinal disc region and resize.

Adapted from: Fundus_IQA_CLI/src/fundus_standardized_process.py

Usage as a module:
    from fundus_preprocess import fundus_image_preprocess_class

    preprocessor = fundus_image_preprocess_class(resize=512)
    img_np = preprocessor.imread('/path/to/fundus.jpg')   # returns RGB numpy array
    img_np = preprocessor.image_crop(img_np)              # returns cropped RGB numpy array

Pass ``do_resize=False`` to crop only (keep the native square-cropped resolution)
so the downstream pipeline's own resize/interpolation is the sole resize step —
useful when a torchvision transform already includes ``transforms.Resize`` and a
second OpenCV resize would be redundant.
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class fundus_image_preprocess_class:
    def __init__(self, resize=512, do_resize=True):
        self.resize_w = resize
        self.resize_h = resize
        self.do_resize = do_resize
        self.tmp_num = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def imread(self, file_path, c=None):
        """Read an image file and return an RGB numpy array."""
        try:
            im = cv2.imread(file_path) if c is None else cv2.imread(file_path, c)
        except Exception:
            img = Image.open(file_path)
            img = np.asanyarray(img)
            im = img[:, :, [2, 1, 0]]

        if im is None:
            raise IOError(f"Cannot read image: {file_path}")

        if im.ndim == 3 and im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def imwrite(self, file_path, image):
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, image)

    def image_crop(self, img, img_name=None):
        """Detect the fundus circle, crop it, and resize to (resize_w, resize_h)."""
        if img is None:
            return img
        resize_w = self.resize_w
        resize_h = self.resize_h
        try:
            r_img, borders, mask, radius = self.process_without_gb(img)

            if (r_img is None) or (borders is None) or (radius is None):
                r_img = img
            else:
                fundus_circle_h = borders[0][1] - borders[0][0]
                fundus_circle_w = borders[0][3] - borders[0][2]
                fundus_full_h   = borders[0][4]
                fundus_full_w   = borders[0][5]
                min_radius = max(fundus_full_h, fundus_full_w) / 4
                max_radius = max(fundus_full_h, fundus_full_w) / 2
                if (fundus_circle_h < fundus_full_h / 2
                        or fundus_circle_w < fundus_full_w / 2
                        or radius < min_radius
                        or radius > max_radius):
                    print(
                        f"Warning: fundus image [{fundus_full_w},{fundus_full_h}]; "
                        f"circle [{fundus_circle_w},{fundus_circle_h}]; radius {radius}"
                    )
                    corrupted_dir = "data/corrupted"
                    if not os.path.isdir(corrupted_dir):
                        os.makedirs(corrupted_dir, exist_ok=True)
                    if img_name is None:
                        img_name = f"crop_fail_{self.tmp_num}.png"
                    self.imwrite(os.path.join(corrupted_dir, img_name), r_img)
                    r_img = img

            if self.do_resize:
                r_img = cv2.resize(r_img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
        except Exception:
            self.tmp_num += 1
            r_img = img
            if self.do_resize:
                r_img = cv2.resize(r_img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
            print(f"Warning: image_crop fallback #{self.tmp_num}, shape {img.shape}")
        return r_img

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def process_without_gb(self, img):
        borders = []
        mask, bbox, center, radius = self.get_mask(img)
        r_img = self.mask_image(img, mask)
        r_img, r_border = self.remove_back_area(r_img, bbox=bbox)
        mask, _ = self.remove_back_area(mask, border=r_border)
        borders.append(r_border)
        r_img, sup_border = self.supplemental_black_area(r_img)
        mask, _ = self.supplemental_black_area(mask, border=sup_border)
        borders.append(sup_border)
        return r_img, borders, (mask * 255).astype(np.uint8), radius

    def get_mask(self, img):
        g_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()
        h, w = g_img.shape
        shape = g_img.shape[:2]
        g_img = cv2.resize(g_img, (0, 0), fx=0.5, fy=0.5)
        tg_img = cv2.normalize(g_img, None, 0, 255, cv2.NORM_MINMAX)
        tmp_mask = self.get_mask_BZ(tg_img)
        center = self._get_center_by_edge(tmp_mask)
        radius = self._get_radius_by_mask_center(tmp_mask, center)
        center = [center[0] * 2, center[1] * 2]
        radius = int(radius * 2)
        s_h = max(0, int(center[0] - radius))
        s_w = max(0, int(center[1] - radius))
        bbox = (s_h, s_w, min(h - s_h, 2 * radius), min(w - s_w, 2 * radius))
        tmp_mask = self._get_circle_by_center_bbox(shape, center, bbox, radius)
        return tmp_mask, bbox, center, radius

    def get_mask_BZ(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        threshold = np.mean(gray_img) / 3 - 5
        _, mask = cv2.threshold(gray_img, max(0, threshold), 1, cv2.THRESH_BINARY)
        nn_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
        new_mask = (1 - mask).astype(np.uint8)
        _, new_mask, _, _ = cv2.floodFill(new_mask, nn_mask, (0, 0), (0), cv2.FLOODFILL_MASK_ONLY)
        _, new_mask, _, _ = cv2.floodFill(
            new_mask, nn_mask,
            (new_mask.shape[1] - 1, new_mask.shape[0] - 1), (0), cv2.FLOODFILL_MASK_ONLY
        )
        mask = mask + new_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        mask = cv2.erode(mask, kernel)
        mask = cv2.dilate(mask, kernel)
        return mask

    def _get_center_by_edge(self, mask):
        center = [0, 0]
        x = mask.sum(axis=1)
        center[0] = np.where(x > x.max() * 0.95)[0].mean()
        x = mask.sum(axis=0)
        center[1] = np.where(x > x.max() * 0.95)[0].mean()
        return center

    def _get_radius_by_mask_center(self, mask, center):
        mask = mask.astype(np.uint8)
        ksize = max(mask.shape[1] // 400 * 2 + 1, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        index = np.where(mask > 0)
        d_int = np.sqrt((index[0] - center[0]) ** 2 + (index[1] - center[1]) ** 2)
        b_count = np.bincount(np.ceil(d_int).astype(np.int_))
        radius = np.where(b_count > b_count.max() * 0.995)[0].max()
        return radius

    def _get_circle_by_center_bbox(self, shape, center, bbox, radius):
        center_mask = np.zeros(shape=shape, dtype='uint8')
        center_tmp = (int(center[0]), int(center[1]))
        center_mask = cv2.circle(center_mask, center_tmp[::-1], int(radius), (1), -1)
        return center_mask

    def mask_image(self, img, mask):
        img[mask <= 0, ...] = 0
        return img

    def remove_back_area(self, img, bbox=None, border=None):
        image = img
        if border is None:
            border = np.array(
                (bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3],
                 img.shape[0], img.shape[1]),
                dtype=np.int_,
            )
        image = image[border[0]:border[1], border[2]:border[3], ...]
        return image, border

    def supplemental_black_area(self, img, border=None):
        image = img
        if border is None:
            h, v = img.shape[:2]
            max_l = max(h, v)
            if image.ndim > 2:
                image = np.zeros((max_l, max_l, img.shape[2]), dtype=img.dtype)
            else:
                image = np.zeros((max_l, max_l), dtype=img.dtype)
            border = (
                int(max_l / 2 - h / 2),
                int(max_l / 2 - h / 2) + h,
                int(max_l / 2 - v / 2),
                int(max_l / 2 - v / 2) + v,
                max_l,
            )
        else:
            max_l = border[4]
            if image.ndim > 2:
                image = np.zeros((max_l, max_l, img.shape[2]), dtype=img.dtype)
            else:
                image = np.zeros((max_l, max_l), dtype=img.dtype)
        image[border[0]:border[1], border[2]:border[3], ...] = img
        return image, border

    def resizeAndPad(self, img, size, padColor=0):
        h, w = img.shape[:2]
        sh, sw = size
        interp = cv2.INTER_AREA if (h > sh or w > sw) else cv2.INTER_CUBIC
        aspect = w / h
        if aspect > 1:
            new_w = sw
            new_h = np.round(new_w / aspect).astype(int)
            pad_vert = (sh - new_h) / 2
            pad_top = np.floor(pad_vert).astype(int)
            pad_bot = np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < 1:
            new_h = sh
            new_w = np.round(new_h * aspect).astype(int)
            pad_horz = (sw - new_w) / 2
            pad_left = np.floor(pad_horz).astype(int)
            pad_right = np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else:
            new_h, new_w = sh, sw
            pad_left = pad_right = pad_top = pad_bot = 0
        if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
            padColor = [padColor] * 3
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(
            scaled_img, pad_top, pad_bot, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=padColor,
        )
        return scaled_img
