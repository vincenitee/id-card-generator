"""
id_pipeline.py -- EMB-CAR ID Card Generator, core pipeline logic.

Reconciled against the confirmed final notebook version. Two real bugs from
that notebook are fixed here (see notes at generate_cards): an undefined
variable reference, and a missing `continue` that silently defeated the
duplicate-print prevention feature.

No Streamlit dependency -- stays reusable/testable standalone. app.py wraps
this with the UI.
"""

import re
import json
import time
import random
import math
import urllib.request
from pathlib import Path
from io import BytesIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import cv2
import requests
from PIL import Image, ImageDraw, ImageFont
from rembg import remove
import barcode
from barcode.writer import ImageWriter
import qrcode
import img2pdf

import gspread
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import AuthorizedSession


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path.cwd()
TEMPLATE_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR / "data"
PHOTOS_DIR = BASE_DIR / "photos"
SIGNATURES_DIR = BASE_DIR / "signatures"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_CARDS_DIR = OUTPUT_DIR / "cards"          # timestamped card PNGs land here
OUTPUT_PRINTABLES_DIR = OUTPUT_DIR / "printables"  # kept for compatibility, currently unused for saving
OUTPUT_FORMS_DIR = OUTPUT_DIR / "forms"          # request form PNGs + PDF land here

FONT_DIR = BASE_DIR / "fonts"
MODELS_DIR = BASE_DIR / "models"
CREDS_PATH = BASE_DIR / "credentials" / "service_account.json"

LAST_RUN_FILE = DATA_DIR / "last_run.json"
PRINTED_LOG_FILE = DATA_DIR / "printed_ids.json"

# Confirmed working shared folder from the final notebook run.
# Change this if the shared folder ever moves.
SHARED_PRINT_DIR = Path(r"C:\ID_Card_Printouts")

SHEET_URL = "https://docs.google.com/spreadsheets/d/1_GKJPfENYbBNBoz11J-eYmQquKm1TTCxu63ChhnLwDM/edit"

MAX_WORKERS = 8
PLACEHOLDER_ID = "EMBB-000-0000"

for _d in [DATA_DIR, PHOTOS_DIR, SIGNATURES_DIR, OUTPUT_DIR, OUTPUT_CARDS_DIR, OUTPUT_PRINTABLES_DIR, OUTPUT_FORMS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# =========================================================
# Module-level state -- shared across calls within this one running process,
# by design: this app runs as a single shared instance on one host.
# =========================================================
_state = {
    "creds": None,
    "gc": None,
    "authed_session": None,
    "haar_cascade": None,
    "df": pd.DataFrame(),
    "all_results": {},
    "run_timestamp": None,
}


# =========================================================
# Retry helper (transient Google API errors)
# =========================================================
def retry_with_backoff(func, max_retries=5, base_delay=1.5, retryable_statuses=(429, 500, 502, 503, 504)):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            is_transient = status in retryable_statuses or isinstance(
                e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
            )
            if not is_transient or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)


# =========================================================
# Fonts -- names kept for continuity, but the actual weight mapping is the
# final swap: font_bold -> Black, font_regular -> SemiBold, font_semibold -> Bold
# =========================================================
def font_bold(size):
    return ImageFont.truetype(str(FONT_DIR / "Montserrat-Black.ttf"), size)

def font_regular(size):
    return ImageFont.truetype(str(FONT_DIR / "Montserrat-SemiBold.ttf"), size)

def font_semibold(size):
    return ImageFont.truetype(str(FONT_DIR / "Montserrat-Bold.ttf"), size)


# =========================================================
# Google auth (lazy -- only connects when actually needed)
# =========================================================
def _get_session():
    if _state["creds"] is None:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        _state["creds"] = Credentials.from_service_account_file(CREDS_PATH, scopes=scopes)
        _state["gc"] = gspread.authorize(_state["creds"])
        _state["authed_session"] = AuthorizedSession(_state["creds"])
    return _state["gc"], _state["authed_session"]


def _get_haar_cascade():
    if _state["haar_cascade"] is None:
        cascade_path = MODELS_DIR / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists() or cascade_path.stat().st_size < 100_000:
            url = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, cascade_path)
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            raise RuntimeError(
                f"Haar cascade failed to load from {cascade_path}. "
                f"Exists: {cascade_path.exists()}, size: {cascade_path.stat().st_size if cascade_path.exists() else 0} bytes."
            )
        _state["haar_cascade"] = cascade
    return _state["haar_cascade"]


YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"


# =========================================================
# Drive fetching
# =========================================================
def extract_drive_file_id(url):
    if not isinstance(url, str) or not url.strip():
        return None
    match = re.search(r'/d/([-\w]{25,})', url)
    if match:
        return match.group(1)
    match = re.search(r'[?&]id=([-\w]{25,})', url)
    if match:
        return match.group(1)
    match = re.search(r'[-\w]{25,}', url)
    return match.group(0) if match else None


def download_drive_image(file_id):
    _, authed_session = _get_session()

    def _do_request():
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        response = authed_session.get(url, timeout=30)
        response.raise_for_status()
        return response

    response = retry_with_backoff(_do_request)
    return Image.open(BytesIO(response.content)).convert("RGB")


# =========================================================
# Photo processing
# =========================================================
def is_background_white(img, tolerance=25, border_pct=0.05):
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    arr = np.array(img_rgb)
    bw, bh = max(1, int(w * border_pct)), max(1, int(h * border_pct))
    border_pixels = np.concatenate([
        arr[:bh, :].reshape(-1, 3), arr[-bh:, :].reshape(-1, 3),
        arr[:, :bw].reshape(-1, 3), arr[:, -bw:].reshape(-1, 3),
    ])
    avg_color = border_pixels.mean(axis=0)
    return np.linalg.norm(np.array([255, 255, 255]) - avg_color) < tolerance


def remove_and_whiten_background(img, max_dim=1024):
    original_size = img.size
    w, h = original_size
    scale = min(1.0, max_dim / max(w, h))
    small_img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS) if scale < 1.0 else img
    small_rgba = remove(small_img.convert("RGBA"))
    img_rgba = small_rgba.resize(original_size, Image.LANCZOS) if scale < 1.0 else small_rgba
    white_bg = Image.new("RGBA", original_size, (255, 255, 255, 255))
    return Image.alpha_composite(white_bg, img_rgba).convert("RGB")


def detect_face_box(img):
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]

    if YUNET_MODEL_PATH.exists():
        detector = cv2.FaceDetectorYN.create(str(YUNET_MODEL_PATH), "", (w, h), score_threshold=0.5)
        _, faces = detector.detect(arr)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: f[-1])
        x, y, bw, bh = best[:4].astype(int)
        return (x, y, x + bw, y + bh)
    else:
        cascade = _get_haar_cascade()
        gray = np.array(img.convert("L"))
        detected = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(detected) == 0:
            return None
        x, y, bw, bh = max(detected, key=lambda f: f[2] * f[3])
        return (x, y, x + bw, y + bh)


def recrop_on_face(img, face_box, target_aspect=190 / 250):
    w, h = img.size
    x_min, y_min, x_max, y_max = face_box
    face_h = y_max - y_min
    face_cx, face_cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    target_h = face_h / 0.5
    target_w = target_h * target_aspect
    crop_top = face_cy - target_h * 0.42
    crop_left = face_cx - target_w / 2
    crop_box = (int(crop_left), int(crop_top), int(crop_left + target_w), int(crop_top + target_h))
    canvas = Image.new("RGB", (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]), (255, 255, 255))
    src_box = (max(0, crop_box[0]), max(0, crop_box[1]), min(w, crop_box[2]), min(h, crop_box[3]))
    paste_pos = (max(0, -crop_box[0]), max(0, -crop_box[1]))
    canvas.paste(img.crop(src_box), paste_pos)
    return canvas


def process_id_photo_from_image(img):
    flags = {"bg_fixed": False, "needs_review": False, "face_detected": False, "cached": False}
    if not is_background_white(img):
        img = remove_and_whiten_background(img)
        flags["bg_fixed"] = True
    face_box = detect_face_box(img)
    if face_box is None:
        flags["needs_review"] = True
    else:
        flags["face_detected"] = True
        w, _ = img.size
        face_cx = (face_box[0] + face_box[2]) / 2
        if abs(face_cx - w / 2) / w > 0.08:
            img = recrop_on_face(img, face_box)
            flags["needs_review"] = True
    return img, flags


def process_signature_image(img, threshold=200):
    gray = img.convert("L")
    arr = np.array(gray)
    alpha = np.where(arr < threshold, 255, 0).astype(np.uint8)
    black_layer = Image.new("RGBA", img.size, (20, 20, 20, 255))
    transparent_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    signature_rgba = Image.composite(black_layer, transparent_layer, Image.fromarray(alpha))
    bbox = signature_rgba.getbbox()
    if bbox:
        signature_rgba = signature_rgba.crop(bbox)
    return signature_rgba


def process_row(row):
    id_num = row["id_number"]
    photo_cache = PHOTOS_DIR / f"{id_num}.png"
    sig_cache = SIGNATURES_DIR / f"{id_num}.png"
    result = {"id_number": id_num, "photo": None, "photo_flags": {}, "signature": None, "signature_error": None}

    if photo_cache.exists():
        result["photo"] = Image.open(photo_cache)
        result["photo_flags"] = {"cached": True}
    else:
        file_id = extract_drive_file_id(row["id_picture_path"])
        if not file_id:
            result["photo_flags"] = {"error": "no valid photo URL/file ID"}
        else:
            try:
                raw_img = download_drive_image(file_id)
                processed_img, flags = process_id_photo_from_image(raw_img)
                processed_img.save(photo_cache)
                result["photo"] = processed_img
                result["photo_flags"] = flags
            except Exception as e:
                result["photo_flags"] = {"error": str(e)}

    if sig_cache.exists():
        result["signature"] = Image.open(sig_cache)
    else:
        sig_file_id = extract_drive_file_id(row["e_signature_path"])
        if not sig_file_id:
            result["signature_error"] = "no valid signature URL/file ID"
        else:
            try:
                raw_sig = download_drive_image(sig_file_id)
                processed_sig = process_signature_image(raw_sig)
                processed_sig.save(sig_cache)
                result["signature"] = processed_sig
            except Exception as e:
                result["signature_error"] = str(e)

    return result


# =========================================================
# Field normalization
# =========================================================
def clean_numeric_id(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    raw_str = str(raw).strip()
    if raw_str == "" or raw_str.upper() in ("N/A", "NAN"):
        return None
    return re.sub(r"\D", "", raw_str)


def normalize_phone(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "N/A", "empty"
    raw_str = str(raw).strip()
    if raw_str == "" or raw_str.upper() in ("N/A", "NAN"):
        return "N/A", "not_provided"
    digits = re.sub(r"\D", "", raw_str)
    if digits.startswith("63") and len(digits) == 12:
        digits = "0" + digits[2:]
    elif digits.startswith("9") and len(digits) == 10:
        digits = "0" + digits
    if digits.startswith("09") and len(digits) == 11:
        return f"{digits[0:4]} {digits[4:7]} {digits[7:11]}", "mobile"
    if 9 <= len(digits) <= 10 and not digits.startswith("09"):
        area_code, local = digits[:-7], digits[-7:]
        return f"({area_code}) {local[:3]} {local[3:]}", "landline"
    if len(digits) == 7:
        return f"{digits[:3]} {digits[3:]}", "landline_no_area_code"
    return digits if digits else raw_str, "unrecognized"


def normalize_tin(raw):
    digits = clean_numeric_id(raw)
    if digits is None:
        return "N/A", "not_provided"
    if len(digits) == 9:
        return f"{digits[0:3]}-{digits[3:6]}-{digits[6:9]}", "tin_individual"
    if len(digits) == 12:
        return f"{digits[0:3]}-{digits[3:6]}-{digits[6:9]}-{digits[9:12]}", "tin_branch"
    return digits, "unrecognized"


def normalize_gsis(raw):
    digits = clean_numeric_id(raw)
    if digits is None:
        return "N/A", "not_provided"
    if len(digits) == 11:
        return f"{digits[0:2]}-{digits[2:9]}-{digits[9:11]}", "gsis_11"
    if len(digits) == 9:
        return f"{digits[0:3]}-{digits[3:6]}-{digits[6:9]}", "gsis_9"
    if len(digits) == 10:
        return f"{digits[0:2]}-{digits[2:8]}-{digits[8:10]}", "gsis_10"
    return digits, "unrecognized"


# =========================================================
# Card compositing helpers
# =========================================================
def cover_resize(img, target_w, target_h):
    img = img.convert("RGB")
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left, top = (new_w - target_w) // 2, (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def fit_resize(img, max_w, max_h):
    """Unlike .thumbnail(), this allows enlarging too -- scales proportionally
    to fit within (max_w, max_h), upscale or downscale as needed."""
    src_w, src_h = img.size
    scale = min(max_w / src_w, max_h / src_h)
    new_w, new_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def wrap_text(text, font, max_width, draw):
    words, lines, line = text.split(), [], ""
    for w in words:
        test = f"{line} {w}".strip()
        if draw.textlength(test, font=font) > max_width and line:
            lines.append(line)
            line = w
        else:
            line = test
    lines.append(line)
    return lines


def draw_centered_multiline(draw, cx, cy, text, font, fill, max_width, line_spacing=4):
    lines = wrap_text(text, font, max_width, draw)
    ascent, descent = font.getmetrics()
    line_height = ascent + descent + line_spacing
    total_height = line_height * len(lines) - line_spacing
    start_y = cy - total_height / 2 + line_height / 2
    for i, line in enumerate(lines):
        y = start_y + i * line_height
        draw.text((cx, y), line, font=font, fill=fill, anchor="mm")


def generate_barcode_image(data, target_w, target_h):
    code128 = barcode.get_barcode_class("code128")
    barcode_obj = code128(data, writer=ImageWriter())
    buf = BytesIO()
    barcode_obj.write(buf, options={"module_height": 8.0, "font_size": 0, "quiet_zone": 1.0, "write_text": False})
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img.resize((target_w, target_h), Image.LANCZOS)


def generate_qr_image(data, size_px):
    """Kept available even though the final card design doesn't use it --
    harmless to leave in case a future revision brings QR back."""
    qr = qrcode.QRCode(border=1, box_size=10)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return img.resize((size_px, size_px), Image.LANCZOS)


# Final measured coordinates.
FRONT_LAYOUT = {
    "photo_box": (110, 168, 537, 620),
    "full_name": (330, 660),
    "position": (330, 733),
    "id_number": (305, 930),
    "barcode_box": (40, 938, 600, 980),
}

BACK_LAYOUT = {
    "address": (64, 100),
    "gsis": (337, 100),
    "tin": (64, 170),
    "blood_type": (337, 170),
    "date_field": (64, 220),
    "emergency_contact_person": (64, 383),
    "emergency_contact_number": (334, 383),
    "signature_box": (150, 815, 495, 925),
}

REQUEST_FORM_LAYOUT = {
    "date": (2160, 525),
    "name": (940, 720),
    "position": (940, 820),
    "address": (940, 920),
    "tin": (940, 1080),
    "gsis": (940, 1170),
    "blood_type": (940, 1260),
    "emergency_contact": (940, 1390),
    "signature_box": (925, 1585, 2355, 1870),
    "photo_2x2_box": (940, 1910, 1539, 2500),
    "photo_1x1_box": (1660, 2070, 1955, 2373),
}


def build_front_card(row, photo, template_path=None):
    template_path = template_path or (TEMPLATE_DIR / "front_template.png")
    card = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(card)
    W = card.width

    px0, py0, px1, py1 = FRONT_LAYOUT["photo_box"]
    if photo:
        card.paste(cover_resize(photo, px1 - px0, py1 - py0), (px0, py0))

    cx = FRONT_LAYOUT["full_name"][0]
    draw.text((cx, FRONT_LAYOUT["full_name"][1]), str(row["full_name"]), font=font_bold(38), fill=(12, 151, 1), anchor="mm")

    position_max_width = W - 80
    draw_centered_multiline(draw, cx, FRONT_LAYOUT["position"][1], str(row["position"]),
                             font=font_bold(20), fill=(255, 255, 255), max_width=position_max_width)

    draw.text((cx, FRONT_LAYOUT["id_number"][1]), str(row["id_number"]), font=font_semibold(13), fill=(16, 31, 187), anchor="mm")

    BARCODE_MARGIN = 26
    _, by0, _, by1 = FRONT_LAYOUT["barcode_box"]
    barcode_w = W - (BARCODE_MARGIN * 2)
    barcode_img = generate_barcode_image(str(row["id_number"]), target_w=barcode_w, target_h=by1 - by0)
    card.paste(barcode_img, (BARCODE_MARGIN, by0))

    return card


def build_back_card(row, signature, template_path=None):
    template_path = template_path or (TEMPLATE_DIR / "back_template.png")
    card = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(card)

    ax, ay = BACK_LAYOUT["address"]
    for i, line in enumerate(wrap_text(str(row["address"]), font_regular(19), 220, draw)):
        draw.text((ax, ay + i * 18), line, font=font_regular(19), fill=(255, 255, 255))

    draw.text(BACK_LAYOUT["gsis"], str(row["gsis"]), font=font_regular(19), fill=(255, 255, 255))
    draw.text(BACK_LAYOUT["tin"], str(row["tin"]), font=font_regular(19), fill=(255, 255, 255))
    draw.text(BACK_LAYOUT["blood_type"], str(row["blood_type"]), font=font_regular(19), fill=(255, 255, 255))
    draw.text(BACK_LAYOUT["emergency_contact_person"], str(row["emergency_contact_person"]), font=font_regular(19), fill=(255, 255, 255))
    draw.text(BACK_LAYOUT["emergency_contact_number"], str(row["emergency_contact_number"]), font=font_regular(19), fill=(255, 255, 255))

    employee_type = str(row.get("employee_type", "")).strip().upper()
    if employee_type == "OJT":
        date_label = "VALID UNTIL"
        date_value = str(row.get("valid_until", "N/A"))
    else:  # REGULAR or JOB ORDER
        date_value = row.get("date_submitted", "N/A")
        date_label = "DATE OF ISSUANCE"
        date_value = date_value.strftime("%B %d, %Y") if hasattr(date_value, "strftime") else str(date_value)

    dx, dy = BACK_LAYOUT["date_field"]
    draw.text((dx, dy), f"{date_label}:", font=font_bold(16), fill=(255, 255, 255))
    draw.text((dx, dy + 20), date_value, font=font_regular(19), fill=(255, 255, 255))

    if signature:
        sx0, sy0, sx1, sy1 = BACK_LAYOUT["signature_box"]
        sig = fit_resize(signature, sx1 - sx0, sy1 - sy0)  # enlarges small signatures too, per your call
        px, py = sx0 + ((sx1 - sx0) - sig.width) // 2, sy0 + ((sy1 - sy0) - sig.height) // 2
        card.paste(sig, (px, py), sig if sig.mode == "RGBA" else None)

    return card


def build_request_form(row, photo, signature, template_path=None):
    template_path = template_path or (TEMPLATE_DIR / "request_form.jpg")
    card = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(card)
    BLACK = (0, 0, 0)

    date_value = row.get("date_submitted", "N/A")
    date_str = date_value.strftime("%m/%d/%Y") if hasattr(date_value, "strftime") else str(date_value)
    draw.text(REQUEST_FORM_LAYOUT["date"], date_str, font=font_semibold(35), fill=BLACK)

    draw.text(REQUEST_FORM_LAYOUT["name"], str(row["full_name"]), font=font_semibold(38), fill=BLACK)
    draw.text(REQUEST_FORM_LAYOUT["position"], str(row["position"]), font=font_semibold(38), fill=BLACK)

    ax, ay = REQUEST_FORM_LAYOUT["address"]
    for i, line in enumerate(wrap_text(str(row["address"]), font_semibold(38), 1400, draw)):
        draw.text((ax, ay + i * 36), line, font=font_semibold(38), fill=BLACK)

    draw.text(REQUEST_FORM_LAYOUT["tin"], str(row["tin"]), font=font_semibold(38), fill=BLACK)
    draw.text(REQUEST_FORM_LAYOUT["gsis"], str(row["gsis"]), font=font_semibold(38), fill=BLACK)
    draw.text(REQUEST_FORM_LAYOUT["blood_type"], str(row["blood_type"]), font=font_semibold(38), fill=BLACK)

    emergency_text = f"{row['emergency_contact_person']} / {row['emergency_contact_number']}"
    draw.text(REQUEST_FORM_LAYOUT["emergency_contact"], emergency_text, font=font_semibold(38), fill=BLACK)

    if photo:
        x0, y0, x1, y1 = REQUEST_FORM_LAYOUT["photo_2x2_box"]
        card.paste(cover_resize(photo, x1 - x0, y1 - y0), (x0, y0))

        x0, y0, x1, y1 = REQUEST_FORM_LAYOUT["photo_1x1_box"]
        card.paste(cover_resize(photo, x1 - x0, y1 - y0), (x0, y0))

    if signature:
        sx0, sy0, sx1, sy1 = REQUEST_FORM_LAYOUT["signature_box"]
        sig = fit_resize(signature, sx1 - sx0, sy1 - sy0)
        px, py = sx0 + ((sx1 - sx0) - sig.width) // 2, sy0 + ((sy1 - sy0) - sig.height) // 2
        card.paste(sig, (px, py), sig if sig.mode == "RGBA" else None)

    return card


def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================================================
# STAGE FUNCTIONS -- called by the UI. Each yields progress strings.
# =========================================================
EXPECTED_RAW_COLUMNS = [
    "Timestamp", "ID NUMBER", "FULL NAME", "POSITION", "HOME ADDRESS",
    "TIN NUMBER (if any)", "GSIS NUMBER (if any)", "BLOOD TYPE",
    "EMERGENCY CONTACT PERSON", "EMERGENCY CONTACT NUMBER",
    "E-SIGNATURE", "ID PICTURE", "EMPLOYEE TYPE",
    "EXPECTED DATE OF COMPLETION",
]


def fetch_data(force_full_reprocess=False):
    yield "Connecting to Google Sheets..."
    gc, _ = _get_session()
    sheet = retry_with_backoff(lambda: gc.open_by_url(SHEET_URL).sheet1)
    records = retry_with_backoff(lambda: sheet.get_all_records())

    if len(records) == 0:
        yield "WARNING: 0 responses in the sheet. Continuing with an empty dataset."
        df = pd.DataFrame(columns=EXPECTED_RAW_COLUMNS)
    else:
        df = pd.DataFrame(records)
    yield f"Loaded {len(df)} responses"

    df = df.rename(columns={
        "Timestamp": "date_submitted", "ID NUMBER": "id_number", "FULL NAME": "full_name",
        "EXPECTED DATE OF COMPLETION": "valid_until",
        "POSITION": "position", "HOME ADDRESS": "address", "TIN NUMBER (if any)": "tin",
        "GSIS NUMBER (if any)": "gsis", "BLOOD TYPE": "blood_type",
        "EMERGENCY CONTACT PERSON": "emergency_contact_person",
        "EMERGENCY CONTACT NUMBER": "emergency_contact_number",
        "E-SIGNATURE": "e_signature_path", "ID PICTURE": "id_picture_path", "EMPLOYEE TYPE": "employee_type",
    })

    df["id_number"] = df["id_number"].astype(str).str.strip()
    df["id_number"] = df["id_number"].replace(["", "nan", "None"], PLACEHOLDER_ID)
    df["id_missing"] = df["id_number"] == PLACEHOLDER_ID

    if len(df) > 0:
        df["emergency_contact_number"] = df["emergency_contact_number"].apply(lambda x: normalize_phone(x)[0])
        df["tin"] = df["tin"].apply(lambda x: normalize_tin(x)[0])
        df["gsis"] = df["gsis"].apply(lambda x: normalize_gsis(x)[0])

    if df["id_missing"].sum() > 0:
        yield f"WARNING: {df['id_missing'].sum()} row(s) using placeholder ID -- review before printing"

    df = df.fillna("N/A")
    df = df.replace(r'^\s*$', "N/A", regex=True)

    exclude_cols = ["e_signature_path", "id_picture_path"]
    for col in [c for c in df.columns if c not in exclude_cols]:
        df[col] = df[col].apply(lambda x: x.upper() if isinstance(x, str) else x)

    _state["df"] = df
    yield "Data cleaned and normalized."

    if len(df) == 0:
        yield "No responses to fetch photos for."
        return

    df["date_submitted"] = pd.to_datetime(df["date_submitted"])
    if LAST_RUN_FILE.exists() and not force_full_reprocess:
        with open(LAST_RUN_FILE) as f:
            last_ts = pd.to_datetime(json.load(f)["last_processed_timestamp"])
        df_to_process = df[df["date_submitted"] > last_ts].copy()
        yield f"Last run: {last_ts} -- fetching {len(df_to_process)} new submission(s)"
    else:
        df_to_process = df.copy()
        yield f"Fetching all {len(df_to_process)} row(s)"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, row): row["id_number"] for _, row in df_to_process.iterrows()}
        for future in as_completed(futures):
            result = future.result()
            _state["all_results"][result["id_number"]] = result
            yield f"{result['id_number']}: photo={result['photo_flags']}, sig_error={result['signature_error']}"

    if len(df_to_process) > 0:
        with open(LAST_RUN_FILE, "w") as f:
            json.dump({"last_processed_timestamp": df_to_process["date_submitted"].max().isoformat()}, f)

    yield f"Done. Fetched {len(df_to_process)} row(s) this run."


def generate_cards(force_reprint_all=False, force_reprint_ids=None):
    """Stage 5. Two bugs from the notebook are fixed here:
    1. 'skipped_count' (undefined) -> 'skipped_placeholder' (the actual counter)
    2. Missing `continue` after the already-printed check -- without it, every
       already-printed row was silently regenerated anyway, defeating the
       whole point of tracking printed_ids in the first place."""
    force_reprint_ids = force_reprint_ids or set()
    df = _state["df"]
    all_results = _state["all_results"]

    if PRINTED_LOG_FILE.exists():
        with open(PRINTED_LOG_FILE) as f:
            printed_ids = set(json.load(f))
    else:
        printed_ids = set()
    yield f"{len(printed_ids)} ID(s) already printed in previous runs"

    run_timestamp = timestamp_str()
    _state["run_timestamp"] = run_timestamp

    generated, skipped_placeholder, skipped_printed = 0, 0, 0
    newly_printed = set()

    for _, row in df.iterrows():
        id_num = row["id_number"]

        if row.get("id_missing", False):
            yield f"Skipping {row['full_name']} -- placeholder ID, needs review"
            skipped_placeholder += 1
            continue

        already_printed = id_num in printed_ids
        force_this_one = force_reprint_all or id_num in force_reprint_ids
        if already_printed and not force_this_one:
            skipped_printed += 1
            continue  # the fix: this was missing, so nothing actually got skipped before

        result = all_results.get(id_num, {})
        photo = result.get("photo") or (Image.open(PHOTOS_DIR / f"{id_num}.png") if (PHOTOS_DIR / f"{id_num}.png").exists() else None)
        signature = result.get("signature") or (Image.open(SIGNATURES_DIR / f"{id_num}.png") if (SIGNATURES_DIR / f"{id_num}.png").exists() else None)

        front = build_front_card(row, photo)
        back = build_back_card(row, signature)
        request_form = build_request_form(row, photo, signature)

        safe_id = str(id_num).replace("/", "-")
        front.save(OUTPUT_CARDS_DIR / f"{safe_id}_front_{run_timestamp}.png", dpi=(300, 300))
        back.save(OUTPUT_CARDS_DIR / f"{safe_id}_back_{run_timestamp}.png", dpi=(300, 300))
        request_form.save(OUTPUT_FORMS_DIR / f"{safe_id}_request_form_{run_timestamp}.png", dpi=(300, 300))
        generated += 1
        newly_printed.add(id_num)
        yield f"Generated: {row['full_name']} ({id_num})"

    printed_ids.update(newly_printed)
    with open(PRINTED_LOG_FILE, "w") as f:
        json.dump(sorted(printed_ids), f)

    yield f"Done. Generated {generated} new card(s). Skipped {skipped_placeholder} placeholder, {skipped_printed} already-printed."


def export_pdf():
    """Stage 6. Reads from OUTPUT_CARDS_DIR (matching where generate_cards
    now saves), builds front-then-back per employee explicitly (not relying
    on filename sort order, which would put 'back' before 'front' alphabetically)."""
    df = _state["df"]
    run_timestamp = _state["run_timestamp"]
    if run_timestamp is None:
        yield "No cards generated yet this session -- run Generate Cards first."
        yield ("__RESULT__", None, None)
        return

    card_files = []
    for _, row in df.iterrows():
        id_num = row["id_number"]
        if row.get("id_missing", False):
            continue
        safe_id = str(id_num).replace("/", "-")
        front_path = OUTPUT_CARDS_DIR / f"{safe_id}_front_{run_timestamp}.png"
        back_path = OUTPUT_CARDS_DIR / f"{safe_id}_back_{run_timestamp}.png"
        if front_path.exists():
            card_files.append(front_path)
        if back_path.exists():
            card_files.append(back_path)

    if not card_files:
        yield "No cards found for this run."
        yield ("__RESULT__", None, None)
        return

    pdf_bytes = img2pdf.convert([str(p) for p in card_files])

    archive_path = OUTPUT_DIR / f"id_cards_print_ready_{run_timestamp}.pdf"
    with open(archive_path, "wb") as f:
        f.write(pdf_bytes)
    yield f"Saved: {archive_path} ({len(card_files)} pages)"

    if SHARED_PRINT_DIR:
        SHARED_PRINT_DIR.mkdir(parents=True, exist_ok=True)
        shared_path = SHARED_PRINT_DIR / f"id_cards_print_ready_{run_timestamp}.pdf"
        with open(shared_path, "wb") as f:
            f.write(pdf_bytes)
        latest_path = SHARED_PRINT_DIR / "LATEST_print_ready.pdf"
        with open(latest_path, "wb") as f:
            f.write(pdf_bytes)
        yield f"Also copied to shared folder: {shared_path}"
        yield f"Updated: {latest_path}"

    yield ("__RESULT__", pdf_bytes, archive_path)


def export_forms_pdf():
    """Stage 6b. Reads from OUTPUT_FORMS_DIR (matching where generate_cards
    saves request forms) and bundles this run's forms into one PDF."""
    df = _state["df"]
    run_timestamp = _state["run_timestamp"]
    if run_timestamp is None:
        yield "No cards generated yet this session -- run Generate Cards first."
        yield ("__RESULT__", None, None)
        return

    form_files = []
    for _, row in df.iterrows():
        id_num = row["id_number"]
        if row.get("id_missing", False):
            continue
        safe_id = str(id_num).replace("/", "-")
        form_path = OUTPUT_FORMS_DIR / f"{safe_id}_request_form_{run_timestamp}.png"
        if form_path.exists():
            form_files.append(form_path)

    if not form_files:
        yield "No request forms found for this run."
        yield ("__RESULT__", None, None)
        return

    form_pdf_bytes = img2pdf.convert([str(p) for p in form_files])

    form_archive_path = OUTPUT_FORMS_DIR / f"request_forms_{run_timestamp}.pdf"
    with open(form_archive_path, "wb") as f:
        f.write(form_pdf_bytes)
    yield f"Saved: {form_archive_path} ({len(form_files)} pages)"

    if SHARED_PRINT_DIR:
        SHARED_PRINT_DIR.mkdir(parents=True, exist_ok=True)
        form_shared_path = SHARED_PRINT_DIR / f"request_forms_{run_timestamp}.pdf"
        with open(form_shared_path, "wb") as f:
            f.write(form_pdf_bytes)
        latest_form_path = SHARED_PRINT_DIR / "LATEST_request_forms.pdf"
        with open(latest_form_path, "wb") as f:
            f.write(form_pdf_bytes)
        yield f"Also copied to shared folder: {form_shared_path}"
        yield f"Updated: {latest_form_path}"

    yield ("__RESULT__", form_pdf_bytes, form_archive_path)