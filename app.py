import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
from collections import Counter, deque
import cv2
import tempfile
import os
import json
import re
import subprocess
import smtplib
import sqlite3
import shutil
import zipfile
import imageio_ffmpeg
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from threading import Event, Lock, Thread
from email.message import EmailMessage
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken

# Configure dashboard
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🤖",
    layout="wide"
)

# Load YOLO only once
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

VEHICLE_CLASSES = ["bicycle", "car", "motorcycle", "bus", "truck"]
LIVE_DETECTION_CLASSES = ["person", *VEHICLE_CLASSES]
PROJECT_DIRECTORY = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIRECTORY / ".env")
HISTORY_FILE = PROJECT_DIRECTORY / "data" / "detection_history.csv"
LOCATION_ALERT_HISTORY_FILE = PROJECT_DIRECTORY / "data" / "location_alert_history.csv"
CAMERA_CONFIG_FILE = PROJECT_DIRECTORY / "data" / "cctv_cameras.json"
CAMERA_KEY_FILE = PROJECT_DIRECTORY / "data" / "cctv_camera.key"
BACKUP_KEY_FILE = PROJECT_DIRECTORY / "data" / "cctv_backup.key"
DATABASE_ERROR_FILE = PROJECT_DIRECTORY / "data" / "database_errors.log"
ALERT_HISTORY_FILE = PROJECT_DIRECTORY / "data" / "alert_history.csv"
DATABASE_FILE = PROJECT_DIRECTORY / "data" / "cctv_database.db"
DEFAULT_RECORDING_DIRECTORY = Path(os.getenv("CCTV_RECORDING_DIRECTORY", r"C:\CCTV_Recordings"))
DEFAULT_BACKUP_DIRECTORY = Path(os.getenv("CCTV_BACKUP_DIRECTORY", r"C:\CCTV_Backup"))
CCTV_MANAGER_CACHE_VERSION = "safety-hardening-v7"
KNOWN_FACES_DIRECTORY = PROJECT_DIRECTORY / "data" / "known_faces"
KNOWN_FACES_FILE = KNOWN_FACES_DIRECTORY / "faces.json"
os.environ.setdefault("DEEPFACE_HOME", str(PROJECT_DIRECTORY))


def boxes_to_dataframe(results, class_names):
    """Convert YOLO boxes into a downloadable results table."""
    rows = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        rows.append({
            "Object": class_names[class_id],
            "Confidence": round(float(box.conf[0]), 4),
            "X1": round(x1, 1),
            "Y1": round(y1, 1),
            "X2": round(x2, 1),
            "Y2": round(y2, 1),
        })
    return pd.DataFrame(rows)


def image_to_png_bytes(image_array):
    output = BytesIO()
    Image.fromarray(image_array).save(output, format="PNG")
    return output.getvalue()


def database_connection():
    DATABASE_FILE.parent.mkdir(exist_ok=True)
    return sqlite3.connect(DATABASE_FILE, timeout=15)


def log_database_error(operation, error):
    DATABASE_ERROR_FILE.parent.mkdir(exist_ok=True)
    try:
        with DATABASE_ERROR_FILE.open("a", encoding="utf-8") as error_log:
            error_log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {operation} | {type(error).__name__}: {error}\n")
    except OSError:
        pass


def log_system_event(category, detail):
    try:
        with database_connection() as connection:
            connection.execute(
                "INSERT INTO system_events (occurred_at, category, detail) VALUES (?, ?, ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), category, detail),
            )
    except sqlite3.Error as error:
        log_database_error("system event", error)


def initialize_cctv_database():
    """Create the local CCTV metadata database. Video stays on disk for performance."""
    with database_connection() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS cameras (
                camera_name TEXT PRIMARY KEY,
                camera_type TEXT,
                network_mode TEXT,
                fixed_location TEXT,
                added_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                file_path TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL,
                file_size_bytes INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS alert_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                camera_name TEXT,
                alert_name TEXT NOT NULL,
                known_faces TEXT,
                location TEXT
            );
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                category TEXT NOT NULL,
                detail TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        connection.execute(
            "INSERT OR REPLACE INTO schema_metadata (key, value) VALUES (?, ?)",
            ("schema_version", "2"),
        )


def sync_cameras_to_database(cameras):
    try:
        with database_connection() as connection:
            for camera in cameras:
                connection.execute("""
                    INSERT INTO cameras (camera_name, camera_type, network_mode, fixed_location, added_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(camera_name) DO UPDATE SET
                        camera_type=excluded.camera_type,
                        network_mode=excluded.network_mode,
                        fixed_location=excluded.fixed_location,
                        updated_at=excluded.updated_at
                """, (
                    camera.get("name", ""), camera.get("type", ""), camera.get("network_mode", ""),
                    camera.get("location", ""), camera.get("added_at", ""), camera.get("updated_at", ""),
                ))
    except sqlite3.Error as error:
        log_database_error("sync cameras", error)


def save_recording_to_database(camera_name, file_path):
    try:
        with database_connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO recordings (camera_name, started_at, file_path, status) VALUES (?, ?, ?, ?)",
                (camera_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(file_path), "Recording"),
            )
    except sqlite3.Error as error:
        log_database_error("save recording", error)


def close_recording_in_database(file_path):
    try:
        path = Path(file_path)
        size = path.stat().st_size if path.exists() else 0
        with database_connection() as connection:
            connection.execute(
                "UPDATE recordings SET ended_at=?, status=?, file_size_bytes=? WHERE file_path=?",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Saved", size, str(path)),
            )
    except (OSError, sqlite3.Error) as error:
        log_database_error("close recording", error)


def remove_recording_from_database(file_path):
    """Keep the recording index consistent after the 30-day retention cleanup."""
    try:
        with database_connection() as connection:
            connection.execute("DELETE FROM recordings WHERE file_path=?", (str(file_path),))
    except sqlite3.Error as error:
        log_database_error("remove recording", error)


def get_backup_cipher():
    BACKUP_KEY_FILE.parent.mkdir(exist_ok=True)
    if not BACKUP_KEY_FILE.exists():
        BACKUP_KEY_FILE.write_bytes(Fernet.generate_key())
    return Fernet(BACKUP_KEY_FILE.read_bytes())


def sqlite_snapshot_bytes():
    """Create a consistent SQLite snapshot even while live threads write new events."""
    temporary_snapshot = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temporary_path = Path(temporary_snapshot.name)
    temporary_snapshot.close()
    try:
        source_connection = database_connection()
        destination_connection = sqlite3.connect(temporary_path)
        try:
            source_connection.backup(destination_connection)
        finally:
            destination_connection.close()
            source_connection.close()
        return temporary_path.read_bytes()
    finally:
        temporary_path.unlink(missing_ok=True)


def create_daily_backup():
    """Create one encrypted, SQLite-consistent metadata backup; large CCTV video stays on disk."""
    try:
        DEFAULT_BACKUP_DIRECTORY.mkdir(parents=True, exist_ok=True)
        backup_file = DEFAULT_BACKUP_DIRECTORY / f"cctv_backup_{datetime.now().strftime('%Y-%m-%d')}.cctvbackup"
        if backup_file.exists():
            return backup_file, "Already created today"
        files_to_backup = [
            ALERT_HISTORY_FILE,
            HISTORY_FILE,
            LOCATION_ALERT_HISTORY_FILE,
            KNOWN_FACES_FILE,
            CAMERA_CONFIG_FILE,
        ]
        archive_bytes = BytesIO()
        with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("data/cctv_database.db", sqlite_snapshot_bytes())
            for source_file in files_to_backup:
                if source_file.exists():
                    archive.write(source_file, source_file.relative_to(PROJECT_DIRECTORY))
        backup_file.write_bytes(get_backup_cipher().encrypt(archive_bytes.getvalue()))
        return backup_file, "Created"
    except (OSError, sqlite3.Error) as error:
        return None, f"Backup error: {error}"


class DailyBackupScheduler:
    """Keeps daily metadata backups current while the Streamlit process remains running."""
    def __init__(self):
        self.stop_event = Event()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            create_daily_backup()
            self.stop_event.wait(60)


@st.cache_resource
def get_daily_backup_scheduler():
    return DailyBackupScheduler()


def restore_backup(backup_file):
    """Restore only known project data files from an encrypted local backup archive."""
    allowed_files = {
        "data/cctv_database.db": DATABASE_FILE,
        "data/alert_history.csv": ALERT_HISTORY_FILE,
        "data/detection_history.csv": HISTORY_FILE,
        "data/location_alert_history.csv": LOCATION_ALERT_HISTORY_FILE,
        "data/known_faces/faces.json": KNOWN_FACES_FILE,
        "data/cctv_cameras.json": CAMERA_CONFIG_FILE,
    }
    try:
        decrypted_archive = get_backup_cipher().decrypt(Path(backup_file).read_bytes())
        with zipfile.ZipFile(BytesIO(decrypted_archive), "r") as archive:
            for archive_name, destination in allowed_files.items():
                if archive_name in archive.namelist():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(archive_name))
        return True, "Backup restored. Restart the dashboard to load restored data."
    except (OSError, zipfile.BadZipFile, InvalidToken) as error:
        return False, f"Restore failed: {error}"


def parse_camera_coordinates(value):
    match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", value or "")
    if not match:
        return None
    latitude, longitude = float(match.group(1)), float(match.group(2))
    if -90 <= latitude <= 90 and -180 <= longitude <= 180:
        return latitude, longitude
    return None


initialize_cctv_database()
DAILY_BACKUP_FILE, DAILY_BACKUP_STATUS = create_daily_backup()
get_daily_backup_scheduler()


class LiveEventLog:
    def __init__(self):
        self.lock = Lock()
        self.events = []

    def add(self, event):
        with self.lock:
            self.events.append(event)
            del self.events[:-100]

    def clear(self):
        with self.lock:
            self.events.clear()

    def snapshot(self):
        with self.lock:
            return list(reversed(self.events))


@st.cache_resource
def get_live_event_log():
    return LiveEventLog()


class AlertSnapshotGallery:
    def __init__(self):
        self.lock = Lock()
        self.snapshots = []

    def add(self, snapshot):
        with self.lock:
            self.snapshots.append(snapshot)
            del self.snapshots[:-12]

    def clear(self):
        with self.lock:
            self.snapshots.clear()

    def snapshot(self):
        with self.lock:
            return list(reversed(self.snapshots))


@st.cache_resource
def get_alert_snapshot_gallery():
    return AlertSnapshotGallery()


class CCTVFeed:
    """Continuously reads one saved network camera without blocking Streamlit."""
    def __init__(self, name, stream_url, recording_settings=None):
        self.name = name
        self.stream_url = stream_url
        self.lock = Lock()
        self.latest_frame = None
        self.error = "Connecting..."
        self.tracking_model = None
        self.tracking_model_label = ""
        self.seen_track_ids = set()
        self.last_sound_time = 0.0
        self.last_snapshot_time = 0.0
        self.last_email_time = 0.0
        self.last_face_request_time = 0.0
        self.last_unknown_face_time = 0.0
        self.first_seen_times = {}
        self.face_worker = None
        self.face_detector = None
        self.recent_face_matches = []
        self.face_confirmation_counts = Counter()
        self.face_activity = []
        self.pre_alert_frames = deque(maxlen=75)
        self.last_pre_alert_frame_at = 0.0
        self.alert_clip_last_frame_at = 0.0
        self.alert_clip_writer = None
        self.alert_clip_file = None
        self.alert_clip_until = 0.0
        self.alert_clip_pending = False
        self.last_alert_clip_time = 0.0
        self.last_face_result_version = -1
        settings = recording_settings or {}
        self.recording_enabled = bool(settings.get("enabled", False))
        self.recording_directory = Path(settings.get("directory", DEFAULT_RECORDING_DIRECTORY))
        self.recording_fps = int(settings.get("fps", 10))
        self.recording_segment_seconds = 30 * 60
        self.recording_writer = None
        self.recording_file = None
        self.recording_started_at = 0.0
        self.recording_last_frame_at = 0.0
        self.recording_last_cleanup_at = 0.0
        self.recording_minimum_free_bytes = 5 * 1024 * 1024 * 1024
        self.recording_mode = settings.get("mode", "Continuous")
        self.motion_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30, detectShadows=False)
        self.last_motion_time = 0.0
        self.status_email_enabled = bool(settings.get("status_email_enabled", False))
        self.camera_location = settings.get("location", "")
        self.connection_state = None
        self.recording_status = "Off"
        self.recording_error = ""
        self.stop_event = Event()
        self.thread = Thread(target=self._read_frames, daemon=True)
        self.thread.start()

    def _read_frames(self):
        camera = None
        while not self.stop_event.is_set():
            if camera is None:
                camera = cv2.VideoCapture(self.stream_url)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not camera.isOpened():
                    self._set_connection_state(False)
                    with self.lock:
                        self.error = "Camera offline. Retrying connection in 3 seconds..."
                    camera.release()
                    camera = None
                    self.stop_event.wait(3)
                    continue
            success, frame = camera.read()
            if success:
                self._set_connection_state(True)
                self._save_pre_alert_frame(frame)
                self._write_alert_clip_frame(frame)
                self._record_frame(frame)
                with self.lock:
                    self.latest_frame = frame
                    self.error = ""
            else:
                self._set_connection_state(False)
                with self.lock:
                    self.error = "Camera stream interrupted. Reconnecting in 3 seconds..."
                camera.release()
                camera = None
                self.stop_event.wait(3)
        if camera is not None:
            camera.release()
        self._close_recording_writer()
        self._close_alert_clip_writer()

    def _save_pre_alert_frame(self, frame):
        now = time.perf_counter()
        if now - self.last_pre_alert_frame_at >= 0.2:
            height, width = frame.shape[:2]
            target_width = min(width, 640)
            target_height = int(height * target_width / width) // 2 * 2
            self.pre_alert_frames.append(cv2.resize(frame, (target_width, target_height)))
            self.last_pre_alert_frame_at = now

    def start_alert_clip(self):
        """Ask the reader thread to save 15 seconds before and 30 seconds after an alert."""
        now = time.perf_counter()
        if now - self.last_alert_clip_time < 60 or self.alert_clip_pending or self.alert_clip_writer is not None:
            return False
        self.alert_clip_pending = True
        self.last_alert_clip_time = now
        return True

    def _close_alert_clip_writer(self):
        if self.alert_clip_writer is not None:
            self.alert_clip_writer.release()
            self.alert_clip_writer = None
        if self.alert_clip_file is not None:
            close_recording_in_database(self.alert_clip_file)
            self.alert_clip_file = None
        self.alert_clip_until = 0.0

    def _write_alert_clip_frame(self, frame):
        now = time.perf_counter()
        try:
            if self.alert_clip_pending:
                folder = self._recording_camera_directory() / "alert_clips"
                folder.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                target_width = min(width, 640)
                target_height = int(height * target_width / width) // 2 * 2
                self.alert_clip_file = folder / f"alert_{datetime.now().strftime('%H-%M-%S')}.mp4"
                writer = cv2.VideoWriter(str(self.alert_clip_file), cv2.VideoWriter_fourcc(*"mp4v"), 5, (target_width, target_height))
                if not writer.isOpened():
                    raise OSError("Windows could not create alert clip")
                self.alert_clip_writer = writer
                for buffered_frame in self.pre_alert_frames:
                    clip_frame = cv2.resize(buffered_frame, (target_width, target_height))
                    self.alert_clip_writer.write(clip_frame)
                save_recording_to_database(self.name, self.alert_clip_file)
                self.alert_clip_until = now + 30
                self.alert_clip_pending = False
                self.recording_status = "Saving alert clip"
            if self.alert_clip_writer is not None:
                if now >= self.alert_clip_until:
                    self._close_alert_clip_writer()
                elif now - self.alert_clip_last_frame_at >= 0.2:
                    height, width = frame.shape[:2]
                    target_width = min(width, 1280)
                    target_height = int(height * target_width / width) // 2 * 2
                    self.alert_clip_writer.write(cv2.resize(frame, (target_width, target_height)))
                    self.alert_clip_last_frame_at = now
        except (OSError, cv2.error) as error:
            self.recording_error = f"Alert clip error: {error}"
            self.alert_clip_pending = False
            self._close_alert_clip_writer()

    def _set_connection_state(self, connected):
        previous_state = self.connection_state
        self.connection_state = connected
        if previous_state is None or previous_state == connected or not self.status_email_enabled:
            return
        status = "Back Online" if connected else "Offline"
        send_camera_status_email_in_background(self.name, status, self.camera_location)

    def add_face_activity(self, status, detail=""):
        """Keep a short, readable diagnostic trail without filling the log every refresh."""
        entry = {"Time": datetime.now().strftime("%H:%M:%S"), "Camera": self.name, "Status": status, "Detail": detail}
        if self.face_activity and self.face_activity[-1]["Status"] == status and self.face_activity[-1]["Detail"] == detail:
            return
        self.face_activity.append(entry)
        del self.face_activity[:-30]

    def face_activity_snapshot(self):
        return list(reversed(self.face_activity))

    def _recording_camera_directory(self):
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", self.name).strip("_") or "camera"
        return self.recording_directory / safe_name / datetime.now().strftime("%Y-%m-%d")

    def _close_recording_writer(self):
        if self.recording_writer is not None:
            self.recording_writer.release()
            self.recording_writer = None
        if self.recording_file is not None:
            close_recording_in_database(self.recording_file)
            self.recording_file = None

    def _cleanup_old_recordings(self):
        """Delete only video files older than the requested retention period."""
        if time.perf_counter() - self.recording_last_cleanup_at < 3600:
            return
        self.recording_last_cleanup_at = time.perf_counter()
        cutoff = time.time() - (30 * 24 * 60 * 60)
        try:
            if self.recording_directory.exists():
                for extension in ("*.mp4", "*.avi"):
                    for recording_file in self.recording_directory.rglob(extension):
                        if recording_file.is_file() and recording_file.stat().st_mtime < cutoff:
                            recording_file.unlink()
                            remove_recording_from_database(recording_file)
                            log_system_event("Retention cleanup", f"Deleted expired recording: {recording_file.name}")
        except OSError as error:
            self.recording_error = f"Could not clean old recordings: {error}"

    def _record_frame(self, frame):
        if not self.recording_enabled:
            return
        now = time.perf_counter()
        if now - self.recording_last_frame_at < 1 / max(1, self.recording_fps):
            return
        self.recording_last_frame_at = now
        try:
            if self.recording_mode == "Motion only":
                small_gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
                foreground = self.motion_detector.apply(small_gray)
                if cv2.countNonZero(foreground) > 900:
                    self.last_motion_time = now
                if now - self.last_motion_time > 10:
                    self.recording_status = "Waiting for motion"
                    self._close_recording_writer()
                    return
            free_bytes = shutil.disk_usage(self.recording_directory if self.recording_directory.exists() else self.recording_directory.parent).free
            if free_bytes < self.recording_minimum_free_bytes:
                self.recording_status = "Paused: less than 5 GB free disk space"
                self.recording_error = "Recording paused to protect the C: drive"
                self._close_recording_writer()
                return
            self._cleanup_old_recordings()
            if self.recording_writer is None or now - self.recording_started_at >= self.recording_segment_seconds:
                self._close_recording_writer()
                camera_directory = self._recording_camera_directory()
                camera_directory.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                target_width = min(width, 1280)
                target_height = int(height * target_width / width)
                target_height -= target_height % 2
                filename = datetime.now().strftime("%H-%M-%S") + ".mp4"
                output_file = camera_directory / filename
                writer = cv2.VideoWriter(
                    str(output_file), cv2.VideoWriter_fourcc(*"mp4v"), self.recording_fps,
                    (target_width, target_height),
                )
                if not writer.isOpened():
                    raise OSError("Windows could not open the MP4 video writer")
                self.recording_writer = writer
                self.recording_file = output_file
                self.recording_started_at = now
                self.recording_status = f"Recording to {camera_directory}"
                self.recording_error = ""
                save_recording_to_database(self.name, output_file)
            height, width = frame.shape[:2]
            target_width = min(width, 1280)
            target_height = int(height * target_width / width)
            target_height -= target_height % 2
            recording_frame = cv2.resize(frame, (target_width, target_height)) if (width, height) != (target_width, target_height) else frame
            self.recording_writer.write(recording_frame)
        except (OSError, cv2.error) as error:
            self.recording_error = f"Recording error: {error}"
            self.recording_status = "Recording stopped"
            self._close_recording_writer()

    def snapshot(self):
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            return frame, self.error

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.recording_error = "Stopping camera reader; finalizing active video when its read operation returns."


class CCTVFeedManager:
    def __init__(self):
        self.lock = Lock()
        self.feeds = {}

    def start(self, camera, recording_settings=None):
        name = camera["name"]
        with self.lock:
            current = self.feeds.get(name)
            settings = recording_settings or {}
            desired_recording = bool(settings.get("enabled", False))
            desired_mode = settings.get("mode", "Continuous")
            desired_fps = int(settings.get("fps", 10))
            desired_status_email = bool(settings.get("status_email_enabled", False))
            if (
                current and current.stream_url == camera["rtsp_url"]
                and current.recording_enabled == desired_recording
                and current.recording_mode == desired_mode
                and current.recording_fps == desired_fps
                and current.status_email_enabled == desired_status_email
            ):
                return
            if current:
                current.stop()
            self.feeds[name] = CCTVFeed(name, camera["rtsp_url"], recording_settings)

    def get(self, name):
        with self.lock:
            return self.feeds.get(name)

    def stop_except(self, camera_names):
        """Prevent previously selected cameras from silently staying active in the background."""
        keep_names = set(camera_names)
        with self.lock:
            names_to_stop = [name for name in self.feeds if name not in keep_names]
            feeds_to_stop = [self.feeds.pop(name) for name in names_to_stop]
        for feed in feeds_to_stop:
            feed.stop()

    def stop_all(self):
        with self.lock:
            feeds = list(self.feeds.values())
            self.feeds.clear()
        for feed in feeds:
            feed.stop()


@st.cache_resource
def get_cctv_feed_manager(cache_version):
    return CCTVFeedManager()


def email_alert_is_configured():
    required_settings = [
        "EMAIL_HOST",
        "EMAIL_PORT",
        "EMAIL_USERNAME",
        "EMAIL_PASSWORD",
        "EMAIL_RECIPIENT",
    ]
    return all(os.getenv(setting) for setting in required_settings)


def get_webrtc_configuration():
    """Use STUN by default and optional authenticated TURN when configured."""
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    turn_url = os.getenv("TURN_URL")
    turn_username = os.getenv("TURN_USERNAME")
    turn_credential = os.getenv("TURN_CREDENTIAL")
    if turn_url and turn_username and turn_credential:
        ice_servers.append({
            "urls": [turn_url],
            "username": turn_username,
            "credential": turn_credential,
        })
    return {"iceServers": ice_servers}


def turn_server_is_configured():
    return all(os.getenv(setting) for setting in ("TURN_URL", "TURN_USERNAME", "TURN_CREDENTIAL"))


def send_email_alert(alert_names, image_bytes, location=""):
    alert_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = EmailMessage()
    message["Subject"] = f"Object Detection Alert: {', '.join(alert_names)}"
    message["From"] = os.environ["EMAIL_USERNAME"]
    message["To"] = os.environ["EMAIL_RECIPIENT"]
    map_link = google_maps_link(location)
    message.set_content(
        f"Detected: {', '.join(alert_names)}"
        f"\nDate and time: {alert_timestamp}"
        + (f"\nLocation: {location.strip()}" if location.strip() else "")
        + (f"\nOpen in Google Maps: {map_link}" if map_link else "")
    )
    message.add_attachment(
        image_bytes,
        maintype="image",
        subtype="png",
        filename="alert_snapshot.png",
    )
    with smtplib.SMTP(os.environ["EMAIL_HOST"], int(os.environ["EMAIL_PORT"])) as server:
        server.starttls()
        server.login(os.environ["EMAIL_USERNAME"], os.environ["EMAIL_PASSWORD"])
        server.send_message(message)


def send_email_alert_in_background(alert_names, image_bytes, event_log, location=""):
    """Keep SMTP work out of the real-time webcam callback."""
    def send():
        try:
            send_email_alert(alert_names, image_bytes, location)
            status = "Sent"
        except (OSError, smtplib.SMTPException):
            status = "Failed"
        event_log.add({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Object": "Email alert",
            "Track ID": "-",
            "Confidence": status,
        })

    Thread(target=send, daemon=True).start()


def send_camera_status_email_in_background(camera_name, status, location=""):
    """Notify only after an established camera connection changes state."""
    if not email_alert_is_configured():
        return
    def send():
        try:
            message = EmailMessage()
            message["Subject"] = f"CCTV Camera {status}: {camera_name}"
            message["From"] = os.environ["EMAIL_USERNAME"]
            message["To"] = os.environ["EMAIL_RECIPIENT"]
            message.set_content(
                f"Camera: {camera_name}\nStatus: {status}\nDate and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                + (f"\nLocation: {location}" if location else "")
            )
            with smtplib.SMTP(os.environ["EMAIL_HOST"], int(os.environ["EMAIL_PORT"])) as server:
                server.starttls()
                server.login(os.environ["EMAIL_USERNAME"], os.environ["EMAIL_PASSWORD"])
                server.send_message(message)
        except (OSError, smtplib.SMTPException):
            pass
    Thread(target=send, daemon=True).start()


def create_detection_pdf(results, class_names, annotated_image, report_title):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image as PdfImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    detection_table = boxes_to_dataframe(results, class_names)
    pdf_output = BytesIO()
    document = SimpleDocTemplate(
        pdf_output,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph(report_title, styles["Title"]),
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"],
        ),
        Spacer(1, 12),
    ]

    summary = [
        ["Total Objects", str(len(detection_table))],
        ["Object Categories", str(detection_table["Object"].nunique() if not detection_table.empty else 0)],
        [
            "Highest Confidence",
            f"{detection_table['Confidence'].max():.0%}" if not detection_table.empty else "N/A",
        ],
    ]
    summary_table = Table(summary, colWidths=[2.2 * inch, 2.2 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1f4e78")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([summary_table, Spacer(1, 14)])

    image_bytes = BytesIO(image_to_png_bytes(annotated_image))
    image_width, image_height = annotated_image.shape[1], annotated_image.shape[0]
    display_width = 6.7 * inch
    display_height = min(display_width * image_height / image_width, 4.2 * inch)
    story.extend([PdfImage(image_bytes, width=display_width, height=display_height), Spacer(1, 14)])

    if not detection_table.empty:
        story.append(Paragraph("Detection Details", styles["Heading2"]))
        report_rows = [["Object", "Confidence", "Coordinates"]]
        for _, row in detection_table.head(25).iterrows():
            report_rows.append([
                row["Object"],
                f"{row['Confidence']:.0%}",
                f"({row['X1']:.0f}, {row['Y1']:.0f}) - ({row['X2']:.0f}, {row['Y2']:.0f})",
            ])
        details_table = Table(report_rows, colWidths=[1.3 * inch, 1.1 * inch, 4.0 * inch])
        details_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e78")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(details_table)

    document.build(story)
    return pdf_output.getvalue()


def save_detection_history(detection_table, source):
    if detection_table.empty:
        return
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    history_rows = detection_table.copy()
    history_rows.insert(0, "Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    history_rows.insert(1, "Source", source)
    history_rows.to_csv(
        HISTORY_FILE,
        mode="a",
        header=not HISTORY_FILE.exists(),
        index=False,
    )


def gps_coordinates_from_location(location):
    """Extract latitude/longitude from the browser GPS label."""
    match = re.search(r"GPS:\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)", location or "")
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def google_maps_link(location):
    coordinates = gps_coordinates_from_location(location)
    if not coordinates:
        return ""
    latitude, longitude = coordinates
    return f"https://www.google.com/maps/search/?api=1&query={latitude:.6f},{longitude:.6f}"


def request_high_accuracy_geolocation(component_key):
    """Request a fresh high-accuracy browser location when hardware GPS is available."""
    from streamlit_js_eval import streamlit_js_eval

    return streamlit_js_eval(
        js_expressions="""new Promise((resolve) => {
            if (!navigator.geolocation) {
                resolve({error: {message: 'Browser does not support geolocation'}});
                return;
            }
            navigator.geolocation.getCurrentPosition(
                (position) => resolve({coords: {
                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude,
                    accuracy: position.coords.accuracy
                }}),
                (error) => resolve({error: {message: error.message}}),
                {enableHighAccuracy: true, timeout: 20000, maximumAge: 0}
            );
        })""",
        key=component_key,
    )


def save_location_alert(name, confidence, location):
    coordinates = gps_coordinates_from_location(location)
    if not coordinates:
        return
    LOCATION_ALERT_HISTORY_FILE.parent.mkdir(exist_ok=True)
    latitude, longitude = coordinates
    history_row = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Name": name,
        "Confidence": confidence,
        "Location": location,
        "Latitude": latitude,
        "Longitude": longitude,
    }])
    history_row.to_csv(
        LOCATION_ALERT_HISTORY_FILE,
        mode="a",
        header=not LOCATION_ALERT_HISTORY_FILE.exists(),
        index=False,
    )


def save_alert_history(camera, alert_names, location, recognized_names=None):
    """Persist meaningful alerts so they can be searched after a monitoring session."""
    if not alert_names:
        return
    ALERT_HISTORY_FILE.parent.mkdir(exist_ok=True)
    row = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Camera": camera,
        "Alert": ", ".join(sorted(set(alert_names))),
        "Known face": ", ".join(sorted(set(recognized_names or []))) or "-",
        "Location": location,
    }])
    row.to_csv(ALERT_HISTORY_FILE, mode="a", header=not ALERT_HISTORY_FILE.exists(), index=False)
    try:
        with database_connection() as connection:
            connection.execute(
                "INSERT INTO alert_events (occurred_at, camera_name, alert_name, known_faces, location) VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), camera,
                    ", ".join(sorted(set(alert_names))),
                    ", ".join(sorted(set(recognized_names or []))) or "-", location,
                ),
            )
    except sqlite3.Error:
        pass


def load_alert_history():
    if not ALERT_HISTORY_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(ALERT_HISTORY_FILE)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def is_alert_schedule_active(enabled, start_time, end_time):
    """Return whether alerts are allowed now; correctly supports overnight schedules."""
    if not enabled:
        return True
    current = datetime.now().time()
    if start_time <= end_time:
        return start_time <= current <= end_time
    return current >= start_time or current <= end_time


def load_location_alert_history():
    if not LOCATION_ALERT_HISTORY_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(LOCATION_ALERT_HISTORY_FILE)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def load_cctv_cameras():
    if not CAMERA_CONFIG_FILE.exists():
        return []
    try:
        cameras = json.loads(CAMERA_CONFIG_FILE.read_text(encoding="utf-8"))
        cipher = get_camera_cipher()
        restored = []
        for saved_camera in cameras:
            camera = dict(saved_camera)
            encrypted_url = camera.pop("rtsp_url_encrypted", "")
            if encrypted_url:
                try:
                    camera["rtsp_url"] = cipher.decrypt(encrypted_url.encode("utf-8")).decode("utf-8")
                except (InvalidToken, UnicodeDecodeError):
                    camera["rtsp_url"] = ""
            restored.append(camera)
        return restored
    except (OSError, json.JSONDecodeError, ValueError):
        return []


def save_cctv_cameras(cameras):
    CAMERA_CONFIG_FILE.parent.mkdir(exist_ok=True)
    cipher = get_camera_cipher()
    protected_cameras = []
    for saved_camera in cameras:
        camera = dict(saved_camera)
        stream_url = camera.pop("rtsp_url", "")
        if stream_url:
            camera["rtsp_url_encrypted"] = cipher.encrypt(stream_url.encode("utf-8")).decode("utf-8")
        protected_cameras.append(camera)
    CAMERA_CONFIG_FILE.write_text(json.dumps(protected_cameras, indent=2), encoding="utf-8")
    sync_cameras_to_database(cameras)


def get_camera_cipher():
    CAMERA_KEY_FILE.parent.mkdir(exist_ok=True)
    if not CAMERA_KEY_FILE.exists():
        if CAMERA_CONFIG_FILE.exists():
            raise RuntimeError(
                "Camera encryption key is missing. Restore data/cctv_camera.key from a secure copy before editing cameras."
            )
        CAMERA_KEY_FILE.write_bytes(Fernet.generate_key())
    return Fernet(CAMERA_KEY_FILE.read_bytes())


def mask_camera_url(camera_url):
    if "@" not in camera_url:
        return camera_url
    protocol, credentials_and_host = camera_url.split("://", 1)
    return f"{protocol}://••••@{credentials_and_host.split('@', 1)[1]}"


def show_detection_details(results, class_names, source=None):
    """Display statistics, a results table, category counts, and CSV export."""
    detection_table = boxes_to_dataframe(results, class_names)
    total_objects = len(detection_table)

    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Total Objects", total_objects)
    metric_two.metric("Object Categories", detection_table["Object"].nunique() if total_objects else 0)
    metric_three.metric(
        "Highest Confidence",
        f"{detection_table['Confidence'].max():.0%}" if total_objects else "—"
    )

    if total_objects == 0:
        st.info("No objects matched the selected settings.")
        return

    if source:
        save_detection_history(detection_table, source)

    st.subheader("Detection Details")
    st.dataframe(detection_table, use_container_width=True, hide_index=True)

    counts = detection_table["Object"].value_counts().rename_axis("Object").reset_index(name="Count")
    chart_column, count_column = st.columns(2)
    chart_column.subheader("Objects by Category")
    chart_column.bar_chart(counts.set_index("Object"))
    count_column.subheader("Category Counts")
    count_column.dataframe(counts, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Detection Results (CSV)",
        data=detection_table.to_csv(index=False).encode("utf-8"),
        file_name="detection_results.csv",
        mime="text/csv",
    )


def load_face_detector():
    cascade_path = PROJECT_DIRECTORY / "models" / "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(str(cascade_path))


@st.cache_resource(show_spinner=False)
def load_insightface_detector():
    """Load the advanced local detector only when a basic detector needs help."""
    from insightface.app import FaceAnalysis

    detector = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    detector.prepare(ctx_id=0, det_size=(640, 640))
    return detector


def load_known_faces():
    """Return locally enrolled face embeddings. No data is uploaded."""
    if not KNOWN_FACES_FILE.exists():
        return []
    try:
        return json.loads(KNOWN_FACES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def save_known_faces(faces):
    KNOWN_FACES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    KNOWN_FACES_FILE.write_text(json.dumps(faces), encoding="utf-8")


def face_embedding(image_bgr, enforce_detection=True):
    from deepface import DeepFace

    representation = DeepFace.represent(
        img_path=image_bgr,
        model_name="Facenet512",
        detector_backend="opencv" if enforce_detection else "skip",
        enforce_detection=enforce_detection,
        align=enforce_detection,
    )
    return np.asarray(representation[0]["embedding"], dtype=np.float32)


def match_known_face(face_bgr, known_faces, minimum_similarity=0.72):
    """Match a detected face against locally registered people only."""
    if not known_faces:
        return None
    try:
        # Align the face first. This produces more stable embeddings when the
        # person has changed hairstyle, facial expression, lighting, or angle.
        embedding = face_embedding(face_bgr, enforce_detection=True)
    except Exception:
        try:
            # A tightly cropped CCTV face may not be detected a second time;
            # retain a local-only fallback for those frames.
            embedding = face_embedding(face_bgr, enforce_detection=False)
        except Exception:
            return None
    embedding_norm = np.linalg.norm(embedding)
    if not embedding_norm:
        return None
    best_match = None
    best_score = -1.0
    for known_face in known_faces:
        reference = np.asarray(known_face.get("embedding", []), dtype=np.float32)
        reference_norm = np.linalg.norm(reference)
        if reference.shape != embedding.shape or not reference_norm:
            continue
        similarity = float(np.dot(embedding, reference) / (embedding_norm * reference_norm))
        if similarity > best_score:
            best_match, best_score = known_face, similarity
    if best_match and best_score >= minimum_similarity:
        return best_match["name"], best_score
    return None


def recognize_known_faces_in_frame(image_bgr, known_faces, minimum_similarity=0.72):
    """Annotate consented, registered faces found in one image or video frame."""
    if not known_faces:
        return image_bgr, []

    detector = load_face_detector()
    if detector.empty():
        return image_bgr, []

    annotated = image_bgr.copy()
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(60, 60),
    )
    matches = []
    for x, y, width, height in faces:
        face_crop = image_bgr[y:y + height, x:x + width]
        match = match_known_face(face_crop, known_faces, minimum_similarity)
        if not match:
            continue
        name, similarity = match
        matches.append({
            "Name": name,
            "Similarity": round(similarity, 3),
            "Face location": f"{x}, {y}",
        })
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 215, 255), 2)
        cv2.putText(
            annotated,
            f"Known: {name} ({similarity:.0%})",
            (x, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 215, 255),
            2,
        )
    return annotated, matches


def detect_low_quality_faces(image_bgr):
    """Find visible faces in small or lower-quality images without identifying them."""
    detector = load_face_detector()
    if detector.empty():
        return []
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grayscale)
    faces = detector.detectMultiScale(
        enhanced,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(35, 35),
    )
    if len(faces):
        return [tuple(map(int, face)) for face in faces]

    # Upscaling helps the detector see a small face, but a result remains a
    # face location only — it is never used as an identity decision.
    enlarged = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    faces = detector.detectMultiScale(
        enlarged,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(40, 40),
    )
    enlarged_faces = [
        (int(x / 2), int(y / 2), int(width / 2), int(height / 2))
        for x, y, width, height in faces
    ]
    if enlarged_faces:
        return enlarged_faces

    # SCRFD/InsightFace is a stronger fallback for side-facing or low-detail
    # faces. If its model cannot be loaded, preserve the fast local fallback.
    try:
        insightface_detector = load_insightface_detector()
        insightface_faces = insightface_detector.get(image_bgr)
        return [
            (
                int(face.bbox[0]),
                int(face.bbox[1]),
                int(face.bbox[2] - face.bbox[0]),
                int(face.bbox[3] - face.bbox[1]),
            )
            for face in insightface_faces
            if face.bbox[2] > face.bbox[0] and face.bbox[3] > face.bbox[1]
        ]
    except Exception:
        return []


def draw_face_locations(image_bgr, face_locations):
    """Draw non-identifying face locations on an already annotated image."""
    annotated = image_bgr.copy()
    for x, y, width, height in face_locations:
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (255, 170, 0), 2)
        cv2.putText(
            annotated,
            "Face detected",
            (x, min(annotated.shape[0] - 8, y + height + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 170, 0),
            2,
        )
    return annotated


def validate_registration_photo(image_bgr, allow_low_quality=False):
    """Return a clear reason when an enrollment photo is unsuitable for matching."""
    detector = load_face_detector()
    if detector.empty():
        return "Face detector is unavailable."
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(35, 35) if allow_low_quality else (60, 60),
    )
    if len(faces) == 0:
        return "No clear face was found."
    if len(faces) > 1:
        return "Use a photo with only one face."
    _, _, width, height = faces[0]
    if min(width, height) < 90 and not allow_low_quality:
        return "Move closer to the camera; the face is too small."
    face_crop = gray[faces[0][1]:faces[0][1] + height, faces[0][0]:faces[0][0] + width]
    brightness = float(face_crop.mean())
    if brightness < 45 and not allow_low_quality:
        return "The face is too dark; use better lighting."
    if brightness > 225 and not allow_low_quality:
        return "The face is too bright; avoid strong backlight or flash."
    sharpness = float(cv2.Laplacian(face_crop, cv2.CV_64F).var())
    if sharpness < 55 and not allow_low_quality:
        return "The face is too blurry; use a sharper photo."
    return None


def enhance_low_light_image(image_bgr):
    """Brighten dark face frames while preserving usable face detail."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) >= 85:
        return image_bgr
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


class KnownFaceRecognitionWorker:
    """Runs expensive face matching away from the live video callback."""
    def __init__(self):
        self.lock = Lock()
        self.busy = False
        self.latest_match = None
        self.result_version = 0

    def request(self, face_bgr, known_faces, center, minimum_similarity=0.72):
        with self.lock:
            if self.busy:
                return
            self.busy = True

        def run_match():
            match = match_known_face(face_bgr.copy(), known_faces, minimum_similarity)
            with self.lock:
                self.latest_match = (*match, *center) if match else None
                self.result_version += 1
                self.busy = False

        Thread(target=run_match, daemon=True).start()

    def get_latest_match(self):
        with self.lock:
            return self.latest_match

    def is_busy(self):
        with self.lock:
            return self.busy

    def get_latest_result(self):
        with self.lock:
            return self.latest_match, self.result_version, self.busy


def class_ids_for_names(class_names, selected_names):
    return [
        class_id
        for class_id, name in class_names.items()
        if name in selected_names
    ]


def show_zoomable_image(image, caption, key):
    """Display an image with controls to magnify and pan its visible area."""
    image_width, image_height = image.size
    st.image(image, caption=caption, use_container_width=True)
    with st.expander("Zoom and Pan Image"):
        zoom_level = st.slider(
            "Zoom level",
            min_value=1.0,
            max_value=4.0,
            value=1.0,
            step=0.25,
            key=f"{key}_zoom",
        )
        crop_width = max(1, int(image_width / zoom_level))
        crop_height = max(1, int(image_height / zoom_level))
        max_x = image_width - crop_width
        max_y = image_height - crop_height

        if max_x or max_y:
            pan_x = st.slider(
                "Pan horizontally",
                min_value=0,
                max_value=max_x,
                value=max_x // 2,
                key=f"{key}_pan_x",
            )
            pan_y = st.slider(
                "Pan vertically",
                min_value=0,
                max_value=max_y,
                value=max_y // 2,
                key=f"{key}_pan_y",
            )
            image = image.crop((pan_x, pan_y, pan_x + crop_width, pan_y + crop_height))

        st.image(image, caption=f"Zoomed view — {zoom_level:.2f}x", use_container_width=True)

# Main heading
st.title("🤖 Real-Time Object Detection Dashboard")

st.write("AI-powered object detection using YOLO and Streamlit.")

st.divider()

# Sidebar navigation
st.sidebar.title("Project Menu")

theme_mode = st.sidebar.selectbox(
    "Dashboard Theme",
    ["Light", "Dark"],
    key="dashboard_theme",
)

if theme_mode == "Dark":
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; color: #f5f7fa; }
        [data-testid="stSidebar"] { background-color: #161b22; }
        [data-testid="stSidebar"] * { color: #f5f7fa; }
        [data-testid="stMetric"] { background-color: #1f2937; border-radius: 10px; padding: 12px; }
        [data-testid="stDataFrame"] { background-color: #161b22; }
        </style>
        """,
        unsafe_allow_html=True,
    )

model_label = st.sidebar.selectbox(
    "Detection Model",
    ["YOLO11 Nano (Fast)", "YOLO11 Small (Balanced)", "YOLO11 Medium (Accurate)"],
    help="Larger models can improve accuracy but require more processing power.",
)
model_paths = {
    "YOLO11 Nano (Fast)": "yolo11n.pt",
    "YOLO11 Small (Balanced)": "yolo11s.pt",
    "YOLO11 Medium (Accurate)": "yolo11m.pt",
}
model = load_model(model_paths[model_label])

option = st.sidebar.selectbox(
    "Select an option",
    [
        "Home",
        "Detection History",
        "Alert Map & Location History",
        "Alert History Search",
        "CCTV Database",
        "System Health",
        "Recorded Video Playback",
        "Camera Test",
        "Incident Timeline",
        "Live Camera Map",
        "Multi-CCTV Camera Manager",
        "Live Multi-CCTV Detection",
        "Object Detection",
        "Vehicle Detection",
        "Face and Emotion Detection",
        "Known Face Registration",
        "Facial Symmetry Scan",
        "3D Face Mesh and Head Pose",
        "Live Webcam Detection",
        "Object Tracking",
    ]
)

# HOME PAGE
if option == "Home":

    st.header("Project Overview")

    st.success("AI Dashboard Successfully Running!")

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Model", "YOLO11")
    col2.metric("Language", "Python")
    col3.metric("Tracking", "ByteTrack")

# DETECTION HISTORY PAGE
elif option == "Detection History":

    st.header("Detection History")
    st.write("Review and download results saved from image and vehicle detections.")

    if not HISTORY_FILE.exists():
        st.info("No detection history has been saved yet.")
    else:
        history_table = pd.read_csv(HISTORY_FILE)
        history_table["Timestamp"] = pd.to_datetime(history_table["Timestamp"])
        start_date, end_date = st.date_input(
            "Filter by date",
            value=(history_table["Timestamp"].min().date(), history_table["Timestamp"].max().date()),
        )
        selected_history_objects = st.multiselect(
            "Filter by object",
            options=sorted(history_table["Object"].dropna().unique()),
            help="Leave empty to show every detected object.",
        )
        filtered_history = history_table[
            history_table["Timestamp"].dt.date.between(start_date, end_date)
        ]
        if selected_history_objects:
            filtered_history = filtered_history[
                filtered_history["Object"].isin(selected_history_objects)
            ]
        st.metric("Saved Detections", len(filtered_history))
        st.dataframe(filtered_history, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Detection History (CSV)",
            data=filtered_history.to_csv(index=False).encode("utf-8"),
            file_name="detection_history.csv",
            mime="text/csv",
        )

# OBJECT DETECTION PAGE
elif option == "Alert Map & Location History":

    st.header("Alert Map & Location History")
    st.write("Review GPS locations saved when a registered face was recognized during an alert.")
    location_history = load_location_alert_history()
    if location_history.empty:
        st.info("No GPS known-face alerts have been saved yet. Enable browser GPS before starting live detection.")
    else:
        location_history["Timestamp"] = pd.to_datetime(location_history["Timestamp"])
        history_start, history_end = st.date_input(
            "Filter location alerts by date",
            value=(location_history["Timestamp"].min().date(), location_history["Timestamp"].max().date()),
            key="location_history_dates",
        )
        names = sorted(location_history["Name"].dropna().unique())
        selected_names = st.multiselect(
            "Filter by registered person",
            options=names,
            key="location_history_names",
        )
        filtered_location_history = location_history[
            location_history["Timestamp"].dt.date.between(history_start, history_end)
        ]
        if selected_names:
            filtered_location_history = filtered_location_history[
                filtered_location_history["Name"].isin(selected_names)
            ]
        st.metric("Saved GPS Alerts", len(filtered_location_history))
        if not filtered_location_history.empty:
            st.map(
                filtered_location_history,
                latitude="Latitude",
                longitude="Longitude",
                size=80,
            )
            location_history_display = filtered_location_history.copy()
            location_history_display["Google Maps"] = location_history_display.apply(
                lambda row: google_maps_link(f"GPS: {row['Latitude']}, {row['Longitude']}"),
                axis=1,
            )
            st.dataframe(
                location_history_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Google Maps": st.column_config.LinkColumn("Open location in Google Maps"),
                },
            )
            st.download_button(
                "Download Location Alert History (CSV)",
                data=filtered_location_history.to_csv(index=False).encode("utf-8"),
                file_name="location_alert_history.csv",
                mime="text/csv",
            )
        else:
            st.info("No saved GPS alerts match the selected filters.")

elif option == "Multi-CCTV Camera Manager":

    st.header("Multi-CCTV Camera Manager")
    st.write(
        "Add each CCTV or phone IP camera once with its name, fixed installation location, and stream address. "
        "Alerts will use the camera's saved location rather than an approximate browser GPS point."
    )
    st.info("Camera stream passwords are encrypted locally on this laptop. Do not share them in chat.")

    show_new_camera_address = st.checkbox("Show stream address while typing", key="show_new_camera_address")
    with st.form("add_cctv_camera", clear_on_submit=True):
        camera_type = st.selectbox(
            "Camera type",
            ["IP CCTV Camera (RTSP)", "Phone IP Camera (HTTP or RTSP)"],
        )
        camera_network_mode = st.selectbox(
            "Camera network mode",
            ["Same Wi-Fi / Local Network", "Remote Private VPN"],
            help="Choose Remote Private VPN only when the phone and dashboard are on different networks.",
        )
        camera_name = st.text_input("Camera name", placeholder="Example: Main Gate Camera")
        camera_location = st.text_input("Camera fixed location", placeholder="Example: East Gudladona - Main Gate")
        camera_coordinates = st.text_input("Camera GPS coordinates (optional)", placeholder="Example: 14.465556, 80.379640")
        new_zone_enabled = st.checkbox("Use a saved alert zone for this camera")
        new_zone_x = st.slider("Alert zone horizontal range (%)", 0, 100, (0, 100), disabled=not new_zone_enabled)
        new_zone_y = st.slider("Alert zone vertical range (%)", 0, 100, (0, 100), disabled=not new_zone_enabled)
        camera_url = st.text_input(
            "Camera stream address",
            placeholder="rtsp://camera-ip:554/stream or http://phone-ip:8080/video",
            type="default" if show_new_camera_address else "password",
            help="Turn on 'Show stream address while typing' above to see and verify the full IP address.",
        )
        if camera_type.startswith("Phone") and camera_network_mode == "Same Wi-Fi / Local Network":
            st.caption(
                "Install an IP-camera app on the phone, connect both devices to the same Wi-Fi, "
                "then copy its HTTP/MJPEG or RTSP stream address here."
            )
        if camera_network_mode == "Remote Private VPN":
            st.caption(
                "Set up the same private VPN on the phone and laptop first. Then use the phone's VPN IP address "
                "in the stream URL, for example: http://100.x.x.x:8080/video"
            )
        add_camera = st.form_submit_button("Save CCTV Camera", type="primary")

    if add_camera:
        if not camera_name.strip() or not camera_location.strip() or not camera_url.strip():
            st.warning("Enter a camera name, fixed location, and RTSP stream address.")
        elif not camera_url.strip().lower().startswith(("rtsp://", "http://", "https://")):
            st.warning("Use a camera address beginning with rtsp://, http://, or https://")
        elif camera_coordinates.strip() and not parse_camera_coordinates(camera_coordinates):
            st.warning("GPS coordinates must be latitude, longitude. Example: 14.465556, 80.379640")
        else:
            cameras = [
                camera for camera in load_cctv_cameras()
                if camera.get("name", "").casefold() != camera_name.strip().casefold()
            ]
            cameras.append({
                "name": camera_name.strip(),
                "type": camera_type,
                "network_mode": camera_network_mode,
                "location": camera_location.strip(),
                "coordinates": camera_coordinates.strip(),
                "alert_zone_enabled": new_zone_enabled,
                "alert_zone_x": list(new_zone_x),
                "alert_zone_y": list(new_zone_y),
                "rtsp_url": camera_url.strip(),
                "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            save_cctv_cameras(cameras)
            st.success(f"Saved {camera_name.strip()}. It is ready for the multi-camera monitoring grid.")

    cameras = load_cctv_cameras()
    st.subheader("Saved CCTV Cameras")
    if cameras:
        camera_table = pd.DataFrame([
            {
                "Camera": camera.get("name", ""),
                "Type": camera.get("type", "IP CCTV Camera (RTSP)"),
                "Network": camera.get("network_mode", "Same Wi-Fi / Local Network"),
                "Fixed Location": camera.get("location", ""),
                "RTSP Address": mask_camera_url(camera.get("rtsp_url", "")),
                "Added": camera.get("added_at", ""),
            }
            for camera in cameras
        ])
        st.dataframe(camera_table, use_container_width=True, hide_index=True)
        if st.button("Encrypt Saved Camera Addresses"):
            save_cctv_cameras(cameras)
            st.success("Saved camera addresses are now encrypted locally.")

        st.subheader("Edit Saved Camera")
        camera_to_edit_name = st.selectbox(
            "Choose a camera to edit",
            [camera["name"] for camera in cameras],
            key="camera_to_edit",
        )
        camera_to_edit = next(camera for camera in cameras if camera["name"] == camera_to_edit_name)
        show_edited_camera_address = st.checkbox("Show stream address while editing", key="show_edited_camera_address")
        with st.form("edit_cctv_camera"):
            edited_type = st.selectbox(
                "Camera type",
                ["IP CCTV Camera (RTSP)", "Phone IP Camera (HTTP or RTSP)"],
                index=0 if camera_to_edit.get("type", "").startswith("IP CCTV") else 1,
                key="edited_camera_type",
            )
            edited_network_mode = st.selectbox(
                "Camera network mode",
                ["Same Wi-Fi / Local Network", "Remote Private VPN"],
                index=1 if camera_to_edit.get("network_mode") == "Remote Private VPN" else 0,
                key="edited_camera_network",
            )
            edited_name = st.text_input("Camera name", value=camera_to_edit.get("name", ""), key="edited_camera_name")
            edited_location = st.text_input("Camera fixed location", value=camera_to_edit.get("location", ""), key="edited_camera_location")
            edited_coordinates = st.text_input("Camera GPS coordinates (optional)", value=camera_to_edit.get("coordinates", ""), key="edited_camera_coordinates")
            edited_zone_enabled = st.checkbox("Use a saved alert zone", value=camera_to_edit.get("alert_zone_enabled", False), key="edited_zone_enabled")
            edited_zone_x = st.slider("Alert zone horizontal range (%)", 0, 100, tuple(camera_to_edit.get("alert_zone_x", [0, 100])), disabled=not edited_zone_enabled, key="edited_zone_x")
            edited_zone_y = st.slider("Alert zone vertical range (%)", 0, 100, tuple(camera_to_edit.get("alert_zone_y", [0, 100])), disabled=not edited_zone_enabled, key="edited_zone_y")
            edited_url = st.text_input(
                "RTSP / HTTP stream address",
                value=camera_to_edit.get("rtsp_url", ""),
                type="default" if show_edited_camera_address else "password",
                help="Turn on 'Show stream address while editing' above to see and verify the full IP address.",
                key="edited_camera_url",
            )
            update_camera = st.form_submit_button("Update Camera", type="primary")

        if update_camera:
            if not edited_name.strip() or not edited_location.strip() or not edited_url.strip():
                st.warning("Camera name, location, and stream address are required.")
            elif not edited_url.strip().lower().startswith(("rtsp://", "http://", "https://")):
                st.warning("Use an address beginning with rtsp://, http://, or https://")
            elif edited_coordinates.strip() and not parse_camera_coordinates(edited_coordinates):
                st.warning("GPS coordinates must be latitude, longitude. Example: 14.465556, 80.379640")
            elif any(camera["name"].casefold() == edited_name.strip().casefold() and camera["name"] != camera_to_edit_name for camera in cameras):
                st.warning("Another saved camera already uses that name.")
            else:
                updated_cameras = []
                for camera in cameras:
                    if camera["name"] == camera_to_edit_name:
                        updated_cameras.append({
                            **camera,
                            "name": edited_name.strip(),
                            "type": edited_type,
                            "network_mode": edited_network_mode,
                            "location": edited_location.strip(),
                            "coordinates": edited_coordinates.strip(),
                            "alert_zone_enabled": edited_zone_enabled,
                            "alert_zone_x": list(edited_zone_x),
                            "alert_zone_y": list(edited_zone_y),
                            "rtsp_url": edited_url.strip(),
                            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })
                    else:
                        updated_cameras.append(camera)
                save_cctv_cameras(updated_cameras)
                get_cctv_feed_manager(CCTV_MANAGER_CACHE_VERSION).stop_all()
                st.success("Camera updated. Start camera monitoring again to use the new address.")
                st.rerun()

        remove_camera = st.selectbox("Remove a saved camera", [camera["name"] for camera in cameras])
        if st.button("Remove Selected Camera"):
            save_cctv_cameras([camera for camera in cameras if camera["name"] != remove_camera])
            st.rerun()
    else:
        st.caption("No CCTV cameras are saved yet.")

elif option == "Alert History Search":

    st.header("Alert History Search")
    st.write("Search alerts previously saved from the multi-CCTV monitoring screen.")
    alert_history = load_alert_history()
    if alert_history.empty:
        st.info("No alert history has been saved yet. Start camera monitoring and wait for an alert.")
    else:
        alert_history["Timestamp"] = pd.to_datetime(alert_history["Timestamp"])
        search_col, camera_col, date_col = st.columns(3)
        search_text = search_col.text_input("Search alert or face name")
        camera_filter = camera_col.selectbox("Camera", ["All cameras", *sorted(alert_history["Camera"].dropna().unique())])
        min_date, max_date = alert_history["Timestamp"].min().date(), alert_history["Timestamp"].max().date()
        selected_dates = date_col.date_input("Date range", value=(min_date, max_date))
        filtered_alerts = alert_history.copy()
        if search_text.strip():
            searchable = filtered_alerts[["Alert", "Known face", "Location"]].fillna("").agg(" ".join, axis=1)
            filtered_alerts = filtered_alerts[searchable.str.contains(search_text.strip(), case=False, na=False)]
        if camera_filter != "All cameras":
            filtered_alerts = filtered_alerts[filtered_alerts["Camera"] == camera_filter]
        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            filtered_alerts = filtered_alerts[filtered_alerts["Timestamp"].dt.date.between(selected_dates[0], selected_dates[1])]
        st.dataframe(filtered_alerts, use_container_width=True, hide_index=True)
        st.download_button("Download Filtered Alert History (CSV)", filtered_alerts.to_csv(index=False).encode("utf-8"), "alert_history.csv", "text/csv")

elif option == "CCTV Database":

    st.header("CCTV Database")
    st.write("Local SQLite database for cameras, video-recording index, and alert events.")
    st.caption(
        f"Daily metadata backup: `{DEFAULT_BACKUP_DIRECTORY}` — {DAILY_BACKUP_STATUS}. "
        "Video recordings remain in the separate CCTV recordings folder."
    )
    backup_files = sorted(DEFAULT_BACKUP_DIRECTORY.glob("cctv_backup_*.cctvbackup"), reverse=True) if DEFAULT_BACKUP_DIRECTORY.exists() else []
    if backup_files:
        selected_backup = st.selectbox("Restore from backup", backup_files, format_func=lambda path: path.name)
        if st.button("Restore Selected Backup"):
            restored, restore_message = restore_backup(selected_backup)
            if restored:
                st.success(restore_message)
            else:
                st.error(restore_message)
    else:
        st.caption("No backup files are available yet.")
    try:
        with database_connection() as connection:
            camera_database = pd.read_sql_query("SELECT * FROM cameras ORDER BY camera_name", connection)
            recording_database = pd.read_sql_query("SELECT * FROM recordings ORDER BY id DESC LIMIT 200", connection)
            alert_database = pd.read_sql_query("SELECT * FROM alert_events ORDER BY id DESC LIMIT 200", connection)
        count_one, count_two, count_three = st.columns(3)
        count_one.metric("Saved cameras", len(camera_database))
        count_two.metric("Indexed video files", len(recording_database))
        count_three.metric("Stored alert events", len(alert_database))
        database_tab_one, database_tab_two, database_tab_three = st.tabs(["Cameras", "Video Recordings", "Alert Events"])
        with database_tab_one:
            st.dataframe(camera_database, use_container_width=True, hide_index=True)
        with database_tab_two:
            st.dataframe(recording_database, use_container_width=True, hide_index=True)
            if not recording_database.empty:
                st.download_button("Download Recording Index (CSV)", recording_database.to_csv(index=False).encode("utf-8"), "recording_index.csv", "text/csv")
        with database_tab_three:
            st.dataframe(alert_database, use_container_width=True, hide_index=True)
    except (sqlite3.Error, pd.errors.DatabaseError) as error:
        st.error(f"Could not read the CCTV database: {error}")

elif option == "System Health":

    st.header("System Health")
    st.write("Review camera state, available storage, recent cleanup activity, and database errors.")
    health_manager = get_cctv_feed_manager(CCTV_MANAGER_CACHE_VERSION)
    saved_health_cameras = load_cctv_cameras()
    health_rows = []
    for camera in saved_health_cameras:
        feed = health_manager.get(camera["name"])
        health_rows.append({
            "Camera": camera["name"],
            "Connection": "Online" if feed and feed.connection_state else "Not monitoring / offline",
            "Recording": feed.recording_status if feed else "Off",
            "Last issue": feed.recording_error if feed and feed.recording_error else "—",
        })
    if health_rows:
        st.dataframe(pd.DataFrame(health_rows), use_container_width=True, hide_index=True)
    try:
        storage = shutil.disk_usage(DEFAULT_RECORDING_DIRECTORY if DEFAULT_RECORDING_DIRECTORY.exists() else DEFAULT_RECORDING_DIRECTORY.parent)
        one, two, three = st.columns(3)
        one.metric("Free recording space", f"{storage.free / (1024 ** 3):.1f} GB")
        two.metric("Daily backup", DAILY_BACKUP_STATUS)
        three.metric("Database", "Connected")
        with database_connection() as connection:
            system_events = pd.read_sql_query("SELECT occurred_at, category, detail FROM system_events ORDER BY id DESC LIMIT 30", connection)
        st.subheader("Recent System Events")
        st.dataframe(system_events, use_container_width=True, hide_index=True)
    except (OSError, sqlite3.Error, pd.errors.DatabaseError) as error:
        st.error(f"Could not load system health: {error}")
    st.subheader("Database Error Log")
    if DATABASE_ERROR_FILE.exists():
        st.code(DATABASE_ERROR_FILE.read_text(encoding="utf-8")[-6000:] or "No errors recorded.")
    else:
        st.success("No database errors recorded.")

elif option == "Recorded Video Playback":

    st.header("Recorded Video Playback")
    st.write("Search saved CCTV recordings by camera and date, then play a completed video file.")
    try:
        with database_connection() as connection:
            recordings = pd.read_sql_query(
                "SELECT * FROM recordings WHERE status='Saved' ORDER BY started_at DESC", connection
            )
        if recordings.empty:
            st.info("No completed recordings yet. A video becomes available after its 30-minute segment ends or monitoring is stopped.")
        else:
            recordings = recordings[recordings["file_path"].map(lambda value: Path(value).is_file())].copy()
            if recordings.empty:
                st.info("The database has no video files that are currently available on this computer.")
            else:
                recordings["started_at"] = pd.to_datetime(recordings["started_at"])
                filter_one, filter_two = st.columns(2)
                playback_camera = filter_one.selectbox("Camera", ["All cameras", *sorted(recordings["camera_name"].unique())])
                available_dates = sorted(recordings["started_at"].dt.date.unique(), reverse=True)
                playback_date = filter_two.selectbox("Recording date", ["All dates", *available_dates])
                filtered_recordings = recordings.copy()
                if playback_camera != "All cameras":
                    filtered_recordings = filtered_recordings[filtered_recordings["camera_name"] == playback_camera]
                if playback_date != "All dates":
                    filtered_recordings = filtered_recordings[filtered_recordings["started_at"].dt.date == playback_date]
                filtered_recordings["Size MB"] = (filtered_recordings["file_size_bytes"] / (1024 * 1024)).round(1)
                st.dataframe(filtered_recordings[["camera_name", "started_at", "ended_at", "Size MB", "file_path"]], use_container_width=True, hide_index=True)
                choices = filtered_recordings.index.tolist()
                selected_recording = st.selectbox(
                    "Choose video to play",
                    choices,
                    format_func=lambda index: f"{filtered_recordings.loc[index, 'camera_name']} — {filtered_recordings.loc[index, 'started_at']}",
                )
                selected_file = filtered_recordings.loc[selected_recording, "file_path"]
                st.video(selected_file)
                selected_size = Path(selected_file).stat().st_size
                if selected_size <= 200 * 1024 * 1024:
                    with open(selected_file, "rb") as video_file:
                        st.download_button("Download This Video", video_file.read(), Path(selected_file).name, "video/mp4")
                else:
                    st.caption("This video is larger than 200 MB. Open it directly from the recording folder to avoid slowing the dashboard.")
    except (OSError, sqlite3.Error, pd.errors.DatabaseError) as error:
        st.error(f"Could not load saved video: {error}")

elif option == "Camera Test":

    st.header("Camera Test")
    st.write("Check one camera connection, live frame, object detector, and face detector from one screen.")
    test_cameras = load_cctv_cameras()
    if not test_cameras:
        st.info("Add a camera first in Multi-CCTV Camera Manager.")
    else:
        test_camera_name = st.selectbox("Camera", [camera["name"] for camera in test_cameras], key="camera_test_name")
        if st.button("Run Camera Test", type="primary"):
            test_camera = next(camera for camera in test_cameras if camera["name"] == test_camera_name)
            with st.spinner("Connecting to camera and testing detection..."):
                capture = cv2.VideoCapture(test_camera["rtsp_url"])
                received, test_frame = capture.read()
                capture.release()
                if not received or test_frame is None:
                    st.error("Camera test failed: no frame received. Check the phone app, IP address, and Wi-Fi.")
                else:
                    face_detector = load_face_detector()
                    faces = face_detector.detectMultiScale(cv2.cvtColor(enhance_low_light_image(test_frame), cv2.COLOR_BGR2GRAY), 1.05, 3, minSize=(30, 30))
                    test_results = model(test_frame, conf=0.35, classes=class_ids_for_names(model.names, LIVE_DETECTION_CLASSES), verbose=False)
                    metric_one, metric_two, metric_three = st.columns(3)
                    metric_one.metric("Camera connection", "Passed")
                    metric_two.metric("Live frame", f"{test_frame.shape[1]} × {test_frame.shape[0]}")
                    metric_three.metric("Faces found", len(faces))
                    annotated_test = test_results[0].plot()
                    for x, y, width, height in faces:
                        cv2.rectangle(annotated_test, (x, y), (x + width, y + height), (255, 160, 0), 2)
                    st.image(annotated_test, channels="BGR", caption="Object results with blue face-detection boxes", use_container_width=True)

elif option == "Incident Timeline":

    st.header("Incident Timeline")
    st.write("Review alerts and completed video files together in time order.")
    try:
        with database_connection() as connection:
            timeline_alerts = pd.read_sql_query("SELECT occurred_at, camera_name, alert_name, known_faces, location FROM alert_events", connection)
            timeline_videos = pd.read_sql_query("SELECT started_at, camera_name, file_path, file_size_bytes FROM recordings WHERE status='Saved'", connection)
        alert_rows = pd.DataFrame({
            "Time": timeline_alerts.get("occurred_at", pd.Series(dtype=str)),
            "Camera": timeline_alerts.get("camera_name", pd.Series(dtype=str)),
            "Type": "Alert",
            "Details": timeline_alerts.get("alert_name", pd.Series(dtype=str)).fillna("") + " | " + timeline_alerts.get("known_faces", pd.Series(dtype=str)).fillna(""),
            "Location / file": timeline_alerts.get("location", pd.Series(dtype=str)),
        })
        video_rows = pd.DataFrame({
            "Time": timeline_videos.get("started_at", pd.Series(dtype=str)),
            "Camera": timeline_videos.get("camera_name", pd.Series(dtype=str)),
            "Type": "Video",
            "Details": "Saved recording",
            "Location / file": timeline_videos.get("file_path", pd.Series(dtype=str)),
        })
        timeline = pd.concat([alert_rows, video_rows], ignore_index=True)
        if timeline.empty:
            st.info("No alerts or completed video recordings are available yet.")
        else:
            timeline["Time"] = pd.to_datetime(timeline["Time"])
            timeline = timeline.sort_values("Time", ascending=False)
            filter_camera = st.selectbox("Filter camera", ["All cameras", *sorted(timeline["Camera"].dropna().unique())])
            if filter_camera != "All cameras":
                timeline = timeline[timeline["Camera"] == filter_camera]
            st.dataframe(timeline, use_container_width=True, hide_index=True)
    except (sqlite3.Error, pd.errors.DatabaseError) as error:
        st.error(f"Could not load the incident timeline: {error}")

elif option == "Live Camera Map":

    st.header("Live Camera Map")
    st.write("Map view of saved cameras with GPS coordinates. Edit a camera to add its exact latitude and longitude.")
    map_cameras = load_cctv_cameras()
    map_rows = []
    manager = get_cctv_feed_manager(CCTV_MANAGER_CACHE_VERSION)
    for camera in map_cameras:
        coordinates = parse_camera_coordinates(camera.get("coordinates", ""))
        if coordinates:
            feed = manager.get(camera["name"])
            map_rows.append({
                "lat": coordinates[0], "lon": coordinates[1], "Camera": camera["name"],
                "Location": camera.get("location", ""),
                "Status": "Online" if feed and feed.connection_state else "Offline / not monitoring",
            })
    if map_rows:
        map_data = pd.DataFrame(map_rows)
        st.map(map_data, latitude="lat", longitude="lon")
        st.dataframe(map_data[["Camera", "Location", "Status", "lat", "lon"]], use_container_width=True, hide_index=True)
    else:
        st.info("No camera GPS coordinates saved yet. Use Multi-CCTV Camera Manager → Edit Saved Camera to add them.")

elif option == "Live Multi-CCTV Detection":

    st.header("Live Multi-CCTV Detection")
    st.write("Monitor saved phone or CCTV streams and detect objects in a live grid.")
    saved_cameras = load_cctv_cameras()
    if not saved_cameras:
        st.info("Add a camera first in Multi-CCTV Camera Manager.")
    else:
        camera_names = [camera["name"] for camera in saved_cameras]
        selected_camera_names = st.multiselect(
            "Choose cameras to monitor",
            options=camera_names,
            default=camera_names[:1],
            key="live_cctv_cameras",
        )
        cctv_confidence = st.slider(
            "CCTV detection confidence",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05,
            key="cctv_confidence",
        )
        cctv_classes = st.multiselect(
            "Detect in CCTV streams",
            options=LIVE_DETECTION_CLASSES,
            default=LIVE_DETECTION_CLASSES,
            key="cctv_classes",
        )
        cctv_alert_classes = st.multiselect(
            "Alert when these objects appear in CCTV streams",
            options=LIVE_DETECTION_CLASSES,
            default=["person"],
            key="cctv_alert_classes",
        )
        cctv_alert_confidence = st.slider(
            "CCTV alert confidence",
            min_value=0.10,
            max_value=0.90,
            value=0.60,
            step=0.05,
            key="cctv_alert_confidence",
            disabled=not cctv_alert_classes,
        )
        cctv_face_recognition = st.checkbox(
            "Recognize registered faces in the CCTV frame",
            key="cctv_face_recognition",
            disabled=not load_known_faces(),
        )
        cctv_face_preview = st.checkbox(
            "Show face detection boxes and activity log",
            value=True,
            key="cctv_face_preview",
            help="Shows blue face boxes even before a known-face match is confirmed.",
        )
        cctv_night_mode = st.checkbox(
            "Night mode: enhance low-light detection",
            key="cctv_night_mode",
            help="Brightens dark frames before object and face analysis. Saved recordings keep the original camera image.",
        )
        cctv_face_confirmation_required = st.slider(
            "Known-face confirmation matches",
            min_value=1,
            max_value=3,
            value=2,
            disabled=not cctv_face_recognition,
            help="Use 1 while testing. Use 2 or 3 in normal use to reduce false face matches.",
        )
        cctv_face_similarity = st.slider(
            "Known-face match sensitivity",
            min_value=0.55,
            max_value=0.85,
            value=0.72,
            step=0.01,
            disabled=not cctv_face_recognition,
            help="Lower values recognize more easily but can increase wrong matches. Start at 0.65 for testing.",
        )
        cctv_sound_alert = st.checkbox(
            "Play sound for CCTV alerts",
            value=True,
            key="cctv_sound_alert",
            disabled=not cctv_alert_classes,
        )
        cctv_email_alert = st.checkbox(
            "Send CCTV alert emails with snapshots",
            key="cctv_email_alert",
            disabled=not cctv_alert_classes or not email_alert_is_configured(),
        )
        st.subheader("Continuous Video Recording")
        cctv_record_continuously = st.checkbox(
            "Record selected cameras continuously",
            value=True,
            key="cctv_record_continuously",
            help="Records while camera monitoring is running. Files are split every 30 minutes and kept for 30 days.",
        )
        cctv_recording_fps = st.slider(
            "Recording frame rate",
            min_value=5,
            max_value=15,
            value=10,
            step=1,
            key="cctv_recording_fps",
            disabled=not cctv_record_continuously,
            help="10 FPS is a good balance between usable footage, disk space, and stable live detection.",
        )
        cctv_recording_mode = st.selectbox(
            "Recording mode",
            ["Continuous", "Motion only"],
            disabled=not cctv_record_continuously,
            help="Motion only records when visible movement is detected and keeps 10 seconds after motion ends.",
        )
        cctv_status_email = st.checkbox(
            "Email when a CCTV camera goes offline or returns online",
            disabled=not email_alert_is_configured(),
            help="The first connection does not send an email. Alerts are sent only after a camera changes state.",
        )
        st.caption(
            f"Recording folder: `{DEFAULT_RECORDING_DIRECTORY}` • 30-minute files • automatic 30-day deletion • max 1280px wide"
        )
        control_one, control_two, control_three = st.columns(3)
        cctv_performance_mode = control_one.selectbox(
            "Performance mode",
            ["Fast (recommended)", "Balanced", "Accurate"],
            help="Fast uses the selected model with the 5-second update interval. Choose a Nano model in the sidebar for best speed.",
            key="cctv_performance_mode",
        )
        cctv_dwell_seconds = control_two.slider(
            "Alert only after visible for seconds",
            0, 15, 0,
            help="0 sends an alert immediately. A higher value reduces alerts when an object quickly passes the camera.",
            key="cctv_dwell_seconds",
        )
        cctv_unknown_face_alert = control_three.checkbox(
            "Alert on unknown faces",
            key="cctv_unknown_face_alert",
            disabled=not cctv_face_recognition,
        )
        cctv_schedule_enabled = st.checkbox("Use an alert schedule", key="cctv_schedule_enabled")
        schedule_one, schedule_two = st.columns(2)
        cctv_schedule_start = schedule_one.time_input("Alerts start", value=datetime.strptime("00:00", "%H:%M").time(), disabled=not cctv_schedule_enabled)
        cctv_schedule_end = schedule_two.time_input("Alerts end", value=datetime.strptime("23:59", "%H:%M").time(), disabled=not cctv_schedule_enabled)
        if cctv_schedule_enabled:
            st.caption("Monitoring stays visible outside this time; only sound and email alerts are paused. Overnight schedules are supported.")
        feed_manager = get_cctv_feed_manager(CCTV_MANAGER_CACHE_VERSION)
        start_cctv_monitoring = st.button(
            "Start Selected Camera Monitoring",
            type="primary",
            disabled=not selected_camera_names or not cctv_classes,
        )
        if st.button("Stop All Camera Monitoring"):
            feed_manager.stop_all()
            st.session_state["cctv_monitoring_active"] = False
            st.rerun()

        if start_cctv_monitoring:
            camera_lookup = {camera["name"]: camera for camera in saved_cameras}
            cctv_recording_settings = {
                "enabled": cctv_record_continuously,
                "directory": str(DEFAULT_RECORDING_DIRECTORY),
                "fps": cctv_recording_fps,
                "mode": cctv_recording_mode,
                "status_email_enabled": cctv_status_email,
            }
            feed_manager.stop_except(selected_camera_names)
            for camera_name in selected_camera_names:
                feed_settings = {**cctv_recording_settings, "location": camera_lookup[camera_name]["location"]}
                feed_manager.start(camera_lookup[camera_name], feed_settings)
            st.session_state["cctv_monitoring_active"] = True

        if st.session_state.get("cctv_monitoring_active"):
            cctv_class_ids = class_ids_for_names(model.names, cctv_classes)
            cctv_image_size = {"Fast (recommended)": 480, "Balanced": 640, "Accurate": 960}[cctv_performance_mode]
            selected_cameras = [
                camera for camera in saved_cameras if camera["name"] in selected_camera_names
            ]
            cctv_event_log = get_live_event_log()
            cctv_snapshot_gallery = get_alert_snapshot_gallery()
            known_faces_for_cctv = load_known_faces()

            @st.fragment(run_every=5)
            def show_cctv_grid():
                columns = st.columns(min(2, max(1, len(selected_cameras))))
                for index, camera in enumerate(selected_cameras):
                    with columns[index % len(columns)]:
                        st.subheader(camera["name"])
                        st.caption(f"Location: {camera['location']}")
                        feed = feed_manager.get(camera["name"])
                        if feed is None:
                            st.warning("Camera has not started yet.")
                            continue
                        frame, feed_error = feed.snapshot()
                        if frame is None:
                            st.info(feed_error or "Connecting to camera...")
                            continue
                        if feed.tracking_model is None or feed.tracking_model_label != model_label:
                            feed.tracking_model = YOLO(model_paths[model_label])
                            feed.tracking_model_label = model_label
                        detection_frame = enhance_low_light_image(frame) if cctv_night_mode else frame
                        results = feed.tracking_model.track(
                            detection_frame,
                            persist=True,
                            conf=cctv_confidence,
                            classes=cctv_class_ids,
                            imgsz=cctv_image_size,
                            verbose=False,
                        )
                        annotated_frame = results[0].plot()
                        zone_enabled = camera.get("alert_zone_enabled", False)
                        zone_x = camera.get("alert_zone_x", [0, 100])
                        zone_y = camera.get("alert_zone_y", [0, 100])
                        frame_height, frame_width = annotated_frame.shape[:2]
                        zone_left, zone_right = int(frame_width * zone_x[0] / 100), int(frame_width * zone_x[1] / 100)
                        zone_top, zone_bottom = int(frame_height * zone_y[0] / 100), int(frame_height * zone_y[1] / 100)
                        if zone_enabled:
                            cv2.rectangle(annotated_frame, (zone_left, zone_top), (zone_right, zone_bottom), (255, 0, 255), 2)
                            cv2.putText(annotated_frame, "ALERT ZONE", (zone_left + 5, max(20, zone_top + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
                        alerts = []
                        recognized_names = []
                        unknown_face_detected = False
                        person_alert = False
                        person_detected = False
                        current_time = time.perf_counter()
                        for box in results[0].boxes:
                            class_name = model.names[int(box.cls[0])]
                            confidence = float(box.conf[0])
                            box_x1, box_y1, box_x2, box_y2 = box.xyxy[0].tolist()
                            box_center_x, box_center_y = (box_x1 + box_x2) / 2, (box_y1 + box_y2) / 2
                            in_alert_zone = not zone_enabled or (zone_left <= box_center_x <= zone_right and zone_top <= box_center_y <= zone_bottom)
                            if class_name in cctv_alert_classes and confidence >= cctv_alert_confidence and in_alert_zone:
                                track_id = int(box.id[0]) if box.id is not None else None
                                dwell_key = (class_name, track_id) if track_id is not None else (class_name, round(float(box.xyxy[0][0]) / 50))
                                feed.first_seen_times.setdefault(dwell_key, current_time)
                                if current_time - feed.first_seen_times[dwell_key] >= cctv_dwell_seconds:
                                    alerts.append(class_name)
                            if class_name == "person" and class_name in cctv_alert_classes and confidence >= cctv_alert_confidence:
                                person_alert = True
                            if class_name == "person":
                                person_detected = True
                            if box.id is not None:
                                track_id = int(box.id[0])
                                event_key = (camera["name"], track_id)
                                if event_key not in feed.seen_track_ids:
                                    cctv_event_log.add({
                                        "Time": datetime.now().strftime("%H:%M:%S"),
                                        "Camera": camera["name"],
                                        "Object": class_name,
                                        "Track ID": str(track_id),
                                        "Confidence": f"{confidence:.0%}",
                                        "Location": camera["location"],
                                    })
                                    feed.seen_track_ids.add(event_key)

                        if (cctv_face_preview or cctv_face_recognition) and (cctv_face_recognition or cctv_face_preview):
                            if feed.face_detector is None:
                                feed.face_detector = load_face_detector()
                            if cctv_face_recognition and feed.face_worker is None:
                                feed.face_worker = KnownFaceRecognitionWorker()
                            enhanced_cctv_frame = enhance_low_light_image(detection_frame)
                            gray_frame = cv2.cvtColor(enhanced_cctv_frame, cv2.COLOR_BGR2GRAY)
                            face_regions = feed.face_detector.detectMultiScale(
                                gray_frame, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
                            )
                            if len(face_regions) == 0:
                                feed.add_face_activity("No face detected", "Move closer to the camera and improve lighting")
                            for x, y, width, height in face_regions:
                                face_region = enhanced_cctv_frame[y:y + height, x:x + width]
                                cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (255, 160, 0), 2)
                                cv2.putText(annotated_frame, "Face detected", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2)
                                feed.add_face_activity("Face detected", f"Face region {width} x {height}")
                                if cctv_face_recognition and not feed.face_worker.is_busy() and current_time - feed.last_face_request_time >= 5:
                                    feed.face_worker.request(
                                        face_region, known_faces_for_cctv, (x + width / 2, y + height / 2),
                                        cctv_face_similarity,
                                    )
                                    feed.last_face_request_time = current_time
                                    feed.add_face_activity("Matching started", f"Sensitivity {cctv_face_similarity:.2f}")
                                latest_match, result_version, worker_busy = feed.face_worker.get_latest_result() if cctv_face_recognition else (None, -1, False)
                                new_face_result = cctv_face_recognition and result_version != feed.last_face_result_version
                                if new_face_result:
                                    feed.last_face_result_version = result_version
                                if latest_match and new_face_result:
                                    name, similarity, match_x, match_y = latest_match
                                    if abs(match_x - (x + width / 2)) < max(width, 80) and abs(match_y - (y + height / 2)) < max(height, 80):
                                        feed.face_confirmation_counts[name] += 1
                                        confirmed = feed.face_confirmation_counts[name] >= cctv_face_confirmation_required
                                        if confirmed:
                                            recognized_names.append(name)
                                            feed.add_face_activity("Known face confirmed", f"{name} ({similarity:.0%})")
                                        else:
                                            feed.add_face_activity("Confirming known face", f"{name} ({similarity:.0%})")
                                        cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                                        cv2.putText(
                                            annotated_frame, f"{name} ({similarity:.0%})" if confirmed else f"Confirming {name}...", (x, max(20, y - 8)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                                        )
                                elif cctv_face_recognition and cctv_unknown_face_alert and new_face_result and not worker_busy and current_time - feed.last_unknown_face_time >= 30:
                                    unknown_face_detected = True
                                    feed.add_face_activity("Unknown face", "No registered face matched")
                                    cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (0, 165, 255), 2)
                                    cv2.putText(annotated_frame, "Unknown face", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

                        cv2.putText(
                            annotated_frame,
                            camera["location"],
                            (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 255),
                            2,
                        )
                        if unknown_face_detected:
                            alerts.append("unknown face")
                        if alerts:
                            feed.start_alert_clip()
                            cv2.putText(
                                annotated_frame,
                                f"ALERT: {', '.join(sorted(set(alerts)))}",
                                (15, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                (0, 0, 255),
                                2,
                            )
                            alerts_allowed = is_alert_schedule_active(cctv_schedule_enabled, cctv_schedule_start, cctv_schedule_end)
                            if cctv_sound_alert and alerts_allowed and current_time - feed.last_sound_time >= 3:
                                try:
                                    import winsound
                                    winsound.Beep(1000, 250)
                                except RuntimeError:
                                    pass
                                feed.last_sound_time = current_time
                            if current_time - feed.last_snapshot_time >= 5:
                                encoded, snapshot = cv2.imencode(".png", annotated_frame)
                                if encoded:
                                    cctv_snapshot_gallery.add({
                                        "Time": datetime.now().strftime("%H:%M:%S"),
                                        "Camera": camera["name"],
                                        "Objects": f"{camera['name']}: {', '.join(sorted(set(alerts)))}",
                                        "Image": snapshot.tobytes(),
                                    })
                                    save_alert_history(camera["name"], alerts, camera["location"], recognized_names)
                                    feed.last_snapshot_time = current_time
                            if unknown_face_detected:
                                feed.last_unknown_face_time = current_time
                            if cctv_email_alert and alerts_allowed and current_time - feed.last_email_time >= 60:
                                encoded, email_snapshot = cv2.imencode(".png", annotated_frame)
                                if encoded:
                                    alert_names = sorted(
                                        set(alerts + [f"Known face: {name}" for name in recognized_names])
                                    )
                                    send_email_alert_in_background(
                                        alert_names, email_snapshot.tobytes(), cctv_event_log, camera["location"]
                                    )
                                    feed.last_email_time = current_time
                        st.image(annotated_frame, channels="BGR", use_container_width=True)

            show_cctv_grid()

            st.subheader("Camera Health")
            health_columns = st.columns(min(3, max(1, len(selected_cameras))))
            for index, camera in enumerate(selected_cameras):
                feed = feed_manager.get(camera["name"])
                camera_frame, camera_error = feed.snapshot() if feed else (None, "Not started")
                health_columns[index % len(health_columns)].metric(
                    camera["name"],
                    "Connected" if camera_frame is not None else "Waiting / offline",
                    camera["location"] if camera_frame is not None else (camera_error or "Check stream address"),
                )
                if feed and feed.recording_enabled:
                    health_columns[index % len(health_columns)].caption(
                        feed.recording_error or "● Continuous recording active"
                    )

            st.subheader("Recording Status & Storage")
            try:
                disk_usage = shutil.disk_usage(DEFAULT_RECORDING_DIRECTORY if DEFAULT_RECORDING_DIRECTORY.exists() else DEFAULT_RECORDING_DIRECTORY.parent)
                storage_one, storage_two, storage_three = st.columns(3)
                storage_one.metric("Free recording space", f"{disk_usage.free / (1024 ** 3):.1f} GB")
                storage_two.metric("Used on drive", f"{disk_usage.used / (1024 ** 3):.1f} GB")
                storage_three.metric("Safety limit", "5 GB free")
            except OSError:
                st.caption("Storage information is unavailable.")
            status_rows = []
            for camera in selected_cameras:
                feed = feed_manager.get(camera["name"])
                if feed:
                    status_rows.append({
                        "Camera": camera["name"],
                        "Connection": "Online" if feed.connection_state else "Offline / connecting",
                        "Recording": feed.recording_status,
                        "Current file": Path(feed.recording_file).name if feed.recording_file else "—",
                        "Mode": feed.recording_mode if feed.recording_enabled else "Off",
                        "Issue": feed.recording_error or "—",
                    })
            if status_rows:
                st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

            if cctv_face_preview or cctv_face_recognition:
                st.subheader("Face Recognition Activity Log")
                face_activity_rows = []
                for camera in selected_cameras:
                    feed = feed_manager.get(camera["name"])
                    if feed:
                        face_activity_rows.extend(feed.face_activity_snapshot())
                if face_activity_rows:
                    st.dataframe(pd.DataFrame(face_activity_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("Start monitoring to see face detection and matching activity.")

            @st.fragment(run_every=5)
            def show_cctv_event_log():
                st.subheader("CCTV Live Event Log")
                events = [event for event in cctv_event_log.snapshot() if event.get("Camera")]
                if events:
                    event_table = pd.DataFrame(events)
                    event_table["Track ID"] = event_table["Track ID"].fillna("-").astype(str)
                    st.dataframe(event_table, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download CCTV Event Log (CSV)",
                        data=event_table.to_csv(index=False).encode("utf-8"),
                        file_name="cctv_event_log.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Start camera monitoring to record tracked CCTV events.")

            show_cctv_event_log()

            st.subheader("CCTV Alert Snapshot Gallery")
            snapshots = cctv_snapshot_gallery.snapshot()
            if snapshots:
                snapshot_camera = st.selectbox("Show snapshots from", ["All cameras", *selected_camera_names], key="cctv_snapshot_camera")
                snapshots = [snapshot for snapshot in snapshots if snapshot_camera == "All cameras" or snapshot.get("Camera") == snapshot_camera]
                gallery_columns = st.columns(min(3, len(snapshots)))
                for index, snapshot in enumerate(snapshots[:9]):
                    gallery_columns[index % len(gallery_columns)].image(snapshot["Image"], caption=f"{snapshot.get('Camera', '')} | {snapshot['Time']} | {snapshot['Objects']}")
            else:
                st.caption("Snapshots appear here when an alert is triggered.")
        else:
            st.info("Select a saved camera and click Start Selected Camera Monitoring.")

# OBJECT DETECTION PAGE
elif option == "Object Detection":

    st.header("📸 AI Image Object Detection")

    st.write("Upload an image and configure the AI detection settings.")

    # Confidence slider
    confidence = st.slider(
        "Minimum Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    st.write(f"Selected confidence: {confidence:.0%}")

    # Available object categories
    class_names = model.names

    # Convert class names into a list
    object_names = list(class_names.values())

    # Object filtering
    selected_objects = st.multiselect(
        "Select Objects to Detect",
        options=object_names,
        default=object_names,
        help="Select one or more object categories."
    )

    # Image uploader
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )
    known_faces_for_image = load_known_faces()
    recognize_image_known_faces = st.checkbox(
        "Recognize registered faces in this image",
        key="image_known_face_recognition",
        disabled=not known_faces_for_image,
        help="Matches only people enrolled on the Known Face Registration page.",
    )
    if not known_faces_for_image:
        st.caption("Register a consenting person on the Known Face Registration page to enable face matching.")
    detect_low_quality_faces_in_image = st.checkbox(
        "Detect small or blurry faces (no identity claim)",
        value=True,
        key="low_quality_face_detection_image",
        help="Marks a visible face location even when it is too unclear for reliable face recognition.",
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Original Image")

        show_zoomable_image(image, "Original image", "object_image")

        if st.button("Detect Objects"):

            if not selected_objects:

                st.warning("Please select at least one object category.")

            else:

                # Convert selected names into YOLO class IDs
                selected_ids = [
                    class_id
                    for class_id, name in class_names.items()
                    if name in selected_objects
                ]

                with st.spinner("AI is detecting objects..."):

                    image_array = np.array(image)

                    # Run YOLO with selected settings
                    results = model(
                        image_array,
                        conf=confidence,
                        classes=selected_ids,
                        verbose=False
                    )

                    # Annotated image and optional registered-face matching.
                    annotated_bgr = results[0].plot()
                    known_face_matches = []
                    if recognize_image_known_faces:
                        annotated_bgr, known_face_matches = recognize_known_faces_in_frame(
                            annotated_bgr,
                            known_faces_for_image,
                        )
                    low_quality_face_locations = []
                    if detect_low_quality_faces_in_image:
                        original_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        low_quality_face_locations = detect_low_quality_faces(original_bgr)
                        annotated_bgr = draw_face_locations(annotated_bgr, low_quality_face_locations)

                    # Convert BGR to RGB
                    annotated_image = annotated_bgr[:, :, ::-1]

                    # Count detections
                    total_objects = len(results[0].boxes)

                st.success("Object Detection Completed!")

                st.subheader("Detection Result")

                st.image(
                    annotated_image,
                    channels="RGB",
                    use_container_width=True
                )
                st.download_button(
                    "Download Detection Screenshot",
                    data=image_to_png_bytes(annotated_image),
                    file_name="object_detection.png",
                    mime="image/png",
                )
                st.download_button(
                    "Download Detection PDF Report",
                    data=create_detection_pdf(
                        results,
                        class_names,
                        annotated_image,
                        "Object Detection Report",
                    ),
                    file_name="object_detection_report.pdf",
                    mime="application/pdf",
                )

                show_detection_details(results, class_names, source="Image Detection")
                if detect_low_quality_faces_in_image:
                    st.caption(f"Face locations detected: {len(low_quality_face_locations)}")
                if recognize_image_known_faces:
                    st.subheader("Registered Face Matches")
                    if known_face_matches:
                        st.dataframe(pd.DataFrame(known_face_matches), use_container_width=True, hide_index=True)
                    else:
                        st.info("No registered face was matched in this image.")
# VEHICLE DETECTION PAGE
elif option == "Vehicle Detection":

    st.header("Vehicle Detection")
    st.write(
        "Upload a road, parking, or traffic image. YOLO detects bicycles, cars, "
        "motorcycles, buses, and trucks."
    )

    vehicle_confidence = st.slider(
        "Minimum Vehicle Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )
    vehicle_file = st.file_uploader(
        "Choose a vehicle image",
        type=["jpg", "jpeg", "png"],
        key="vehicle_image",
    )
    known_faces_for_vehicle = load_known_faces()
    recognize_vehicle_known_faces = st.checkbox(
        "Recognize registered faces in this image",
        key="vehicle_known_face_recognition",
        disabled=not known_faces_for_vehicle,
        help="Matches only people enrolled on the Known Face Registration page.",
    )
    if not known_faces_for_vehicle:
        st.caption("Register a consenting person on the Known Face Registration page to enable face matching.")

    if vehicle_file is not None:
        vehicle_image = Image.open(vehicle_file).convert("RGB")
        show_zoomable_image(vehicle_image, "Uploaded image", "vehicle_image")

        if st.button("Detect Vehicles"):
            vehicle_ids = [
                class_id
                for class_id, name in model.names.items()
                if name in VEHICLE_CLASSES
            ]
            start_time = time.perf_counter()
            results = model(
                np.array(vehicle_image),
                conf=vehicle_confidence,
                classes=vehicle_ids,
                verbose=False,
            )
            elapsed = time.perf_counter() - start_time
            annotated_bgr = results[0].plot()
            vehicle_known_face_matches = []
            if recognize_vehicle_known_faces:
                annotated_bgr, vehicle_known_face_matches = recognize_known_faces_in_frame(
                    annotated_bgr,
                    known_faces_for_vehicle,
                )
            annotated_image = annotated_bgr[:, :, ::-1]

            st.subheader("Vehicle Detection Result")
            st.image(annotated_image, channels="RGB", use_container_width=True)
            st.download_button(
                "Download Vehicle Detection Screenshot",
                data=image_to_png_bytes(annotated_image),
                file_name="vehicle_detection.png",
                mime="image/png",
            )
            st.download_button(
                "Download Vehicle Detection PDF Report",
                data=create_detection_pdf(
                    results,
                    model.names,
                    annotated_image,
                    "Vehicle Detection Report",
                ),
                file_name="vehicle_detection_report.pdf",
                mime="application/pdf",
            )
            st.caption(f"Processed in {elapsed:.2f} seconds")
            show_detection_details(results, model.names, source="Vehicle Detection")
            if recognize_vehicle_known_faces:
                st.subheader("Registered Face Matches")
                if vehicle_known_face_matches:
                    st.dataframe(pd.DataFrame(vehicle_known_face_matches), use_container_width=True, hide_index=True)
                else:
                    st.info("No registered face was matched in this image.")

# FACE AND EMOTION DETECTION PAGE
elif option == "Known Face Registration":
    st.header("Known Face Registration")
    st.info(
        "Register only people who have agreed to be enrolled. Face photos and matching data stay in this project's local data folder."
    )
    registered_name = st.text_input("Person's name", key="registered_face_name")
    registered_photos = st.file_uploader(
        "Clear photos with one front-facing face (select 2–4 photos)",
        type=["jpg", "jpeg", "png"],
        key="registered_face_photo",
        accept_multiple_files=True,
    )
    registered_camera_photo = st.camera_input(
        "Or take a clear face photo now",
        key="registered_face_camera_photo",
        help="Use even lighting and keep one face close to the camera.",
    )
    allow_low_quality_registration = st.checkbox(
        "Accept low-light or slightly blurry photos when a face is visible",
        value=True,
        key="allow_low_quality_registration",
        help="The photo is brightened automatically. Recognition accuracy may be lower for unclear photos.",
    )
    if st.button("Register Known Face", type="primary"):
        registration_photos = list(registered_photos or [])
        if registered_camera_photo is not None:
            registration_photos.append(registered_camera_photo)
        if not registered_name.strip() or not registration_photos:
            st.warning("Enter a name and select or take at least one clear face photo.")
        else:
            known_faces = load_known_faces()
            registered_count = 0
            failed_files = []
            with st.spinner("Creating local face match profiles..."):
                for registered_photo in registration_photos:
                    photo_bytes = np.frombuffer(registered_photo.getvalue(), dtype=np.uint8)
                    photo = cv2.imdecode(photo_bytes, cv2.IMREAD_COLOR)
                    if photo is None:
                        failed_files.append(f"{registered_photo.name} (unreadable file)")
                        continue
                    photo_for_registration = enhance_low_light_image(photo)
                    quality_issue = validate_registration_photo(
                        photo_for_registration,
                        allow_low_quality=allow_low_quality_registration,
                    )
                    if quality_issue:
                        failed_files.append(f"{registered_photo.name} ({quality_issue})")
                        continue
                    try:
                        embedding = face_embedding(photo_for_registration, enforce_detection=True)
                        known_faces.append({
                            "name": registered_name.strip(),
                            "embedding": embedding.tolist(),
                            "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })
                        registered_count += 1
                    except Exception:
                        failed_files.append(f"{registered_photo.name} (face profile could not be created)")
            if registered_count:
                save_known_faces(known_faces)
                st.success(f"Added {registered_count} face photo(s) for {registered_name.strip()}.")
            if failed_files:
                st.warning("Not registered: " + "; ".join(failed_files))

    known_faces = load_known_faces()
    st.subheader("Registered People")
    if known_faces:
        face_summary = pd.DataFrame(known_faces).groupby("name", as_index=False).agg(
            Photos=("embedding", "count"), Latest_registration=("registered_at", "max")
        )
        st.dataframe(
            face_summary.rename(columns={"name": "Name", "Latest_registration": "Latest registration"}),
            use_container_width=True,
            hide_index=True,
        )
        remove_name = st.selectbox("Remove a registered person", [face["name"] for face in known_faces])
        if st.button("Remove Selected Person"):
            save_known_faces([face for face in known_faces if face["name"] != remove_name])
            st.rerun()
    else:
        st.caption("No people are registered yet.")

elif option == "Face and Emotion Detection":

    st.header("Face and Emotion Detection")
    st.write(
        "Upload a clear face image. The app locates faces locally, then estimates "
        "visible facial expressions when the optional DeepFace package is installed."
    )
    face_file = st.file_uploader(
        "Choose a face image",
        type=["jpg", "jpeg", "png"],
        key="face_image",
    )

    if face_file is not None:
        face_image = Image.open(face_file).convert("RGB")
        show_zoomable_image(face_image, "Uploaded image", "face_image")

        if st.button("Detect Faces and Emotions"):
            image_array = np.array(face_image)
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            face_detector = cv2.CascadeClassifier(
    os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
)
            faces = face_detector.detectMultiScale(
                gray_image,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(60, 60),
            )
            annotated_image = image_array.copy()
            emotion_rows = []

            try:
                from deepface import DeepFace
                deepface_error = None
            except ImportError as error:
                DeepFace = None
                deepface_error = str(error)

            for face_number, (x, y, width, height) in enumerate(faces, start=1):
                cv2.rectangle(
                    annotated_image,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2,
                )
                row = {
                    "Face": face_number,
                    "X": int(x),
                    "Y": int(y),
                    "Width": int(width),
                    "Height": int(height),
                    "Dominant Emotion": "Unavailable",
                    "Confidence": None,
                }

                if DeepFace is not None:
                    face_crop = image_array[y:y + height, x:x + width]
                    try:
                        analysis = DeepFace.analyze(
                            face_crop,
                            actions=["emotion"],
                            detector_backend="skip",
                            enforce_detection=False,
                        )
                        if isinstance(analysis, list):
                            analysis = analysis[0]
                        emotion = analysis["dominant_emotion"]
                        score = analysis["emotion"][emotion]
                        row["Dominant Emotion"] = emotion.title()
                        row["Confidence"] = round(float(score) / 100, 4)
                        cv2.putText(
                            annotated_image,
                            emotion.title(),
                            (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                    except Exception as error:
                        row["Dominant Emotion"] = f"Analysis failed: {error}"
                emotion_rows.append(row)

            st.subheader("Face Detection Result")
            st.image(annotated_image, channels="RGB", use_container_width=True)
            st.download_button(
                "Download Face Detection Screenshot",
                data=image_to_png_bytes(annotated_image),
                file_name="face_detection.png",
                mime="image/png",
            )
            st.metric("Faces Detected", len(faces))

            if len(faces) == 0:
                st.info("No faces were detected. Try an image with a larger, front-facing face.")
            else:
                st.dataframe(pd.DataFrame(emotion_rows), use_container_width=True, hide_index=True)
                if DeepFace is None:
                    st.warning(
                        "Face locations are available, but emotion estimates could not start: "
                        f"{deepface_error}"
                    )

# FACIAL SYMMETRY SCAN PAGE
elif option == "Facial Symmetry Scan":

    st.header("Facial Symmetry Scan")
    st.write(
        "Upload a clear, front-facing image. The scan compares paired facial "
        "landmarks to show geometric left-right alignment."
    )
    symmetry_file = st.file_uploader(
        "Choose a face image",
        type=["jpg", "jpeg", "png"],
        key="symmetry_image",
    )

    if symmetry_file is not None:
        symmetry_image = Image.open(symmetry_file).convert("RGB")
        show_zoomable_image(symmetry_image, "Uploaded image", "symmetry_image")

        if st.button("Scan Facial Symmetry"):
            try:
                import mediapipe as mp
            except ImportError:
                st.warning(
                    "Install the updated requirements to enable facial symmetry scanning: "
                    "python -m pip install -r requirements.txt"
                )
            else:
                image_array = np.array(symmetry_image)
                paired_landmarks = [
                    (33, 263),
                    (133, 362),
                    (234, 454),
                    (61, 291),
                    (172, 397),
                ]
                model_directory = PROJECT_DIRECTORY / "models"
                face_model_path = model_directory / "face_landmarker.task"

                if not face_model_path.exists():
                    try:
                        from urllib.request import urlretrieve

                        model_directory.mkdir(exist_ok=True)
                        with st.spinner("Downloading the Face Landmarker model..."):
                            urlretrieve(
                                "https://storage.googleapis.com/mediapipe-models/"
                                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
                                face_model_path,
                            )
                    except Exception as error:
                        st.error(f"Unable to download the Face Landmarker model: {error}")

                if not face_model_path.exists():
                    st.stop()

                base_options = mp.tasks.BaseOptions(model_asset_path=str(face_model_path))
                landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                )
                scanner = mp.tasks.vision.FaceLandmarker.create_from_options(
                    landmarker_options
                )
                scan_result = scanner.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
                )
                scanner.close()

                if not scan_result.face_landmarks:
                    st.info("No face was detected. Use a clear, front-facing image with good lighting.")
                else:
                    landmarks = scan_result.face_landmarks[0]
                    image_height, image_width = image_array.shape[:2]
                    points = [
                        (int(point.x * image_width), int(point.y * image_height))
                        for point in landmarks
                    ]
                    face_left = min(point[0] for point in points)
                    face_right = max(point[0] for point in points)
                    face_width = max(face_right - face_left, 1)
                    center_x = points[1][0]
                    annotated_image = image_array.copy()
                    differences = []

                    cv2.line(
                        annotated_image,
                        (center_x, 0),
                        (center_x, image_height),
                        (0, 255, 0),
                        2,
                    )
                    for left_index, right_index in paired_landmarks:
                        left_point = points[left_index]
                        right_point = points[right_index]
                        left_distance = center_x - left_point[0]
                        right_distance = right_point[0] - center_x
                        differences.append(abs(left_distance - right_distance) / face_width)
                        cv2.circle(annotated_image, left_point, 4, (255, 0, 0), -1)
                        cv2.circle(annotated_image, right_point, 4, (255, 0, 0), -1)
                        cv2.line(annotated_image, left_point, right_point, (255, 255, 0), 1)

                    average_difference = float(np.mean(differences))
                    symmetry_score = max(0.0, min(100.0, (1 - average_difference / 0.20) * 100))

                    st.subheader("Symmetry Scan Result")
                    st.image(annotated_image, channels="RGB", use_container_width=True)
                    score_column, difference_column = st.columns(2)
                    score_column.metric("Geometric Symmetry Score", f"{symmetry_score:.1f}%")
                    difference_column.metric("Average Landmark Difference", f"{average_difference:.3f}")
                    st.caption(
                        "The green line is the estimated facial center. Blue points are paired "
                        "landmarks used for comparison. Head angle, expression, lens distortion, "
                        "and lighting can change the score."
                    )

# 3D FACE MESH AND HEAD POSE PAGE
elif option == "3D Face Mesh and Head Pose":

    st.header("3D Face Mesh and Head Pose")
    st.write(
        "Start the camera to display facial landmarks and an estimated head direction. "
        "This feature does not identify people."
    )
    mesh_detail = st.selectbox(
        "Mesh detail",
        ["Full face mesh", "Face outline only"],
        key="mesh_detail",
    )

    try:
        import av
        import mediapipe as mp
        from streamlit_webrtc import webrtc_streamer
    except ImportError:
        st.warning(
            "Install the updated requirements to enable the 3D face mesh feature: "
            "python -m pip install -r requirements.txt"
        )
    else:
        model_directory = PROJECT_DIRECTORY / "models"
        face_model_path = model_directory / "face_landmarker.task"
        if not face_model_path.exists():
            try:
                from urllib.request import urlretrieve

                model_directory.mkdir(exist_ok=True)
                with st.spinner("Downloading the Face Landmarker model..."):
                    urlretrieve(
                        "https://storage.googleapis.com/mediapipe-models/"
                        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
                        face_model_path,
                    )
            except Exception as error:
                st.error(f"Unable to download the Face Landmarker model: {error}")

        if face_model_path.exists():
            base_options = mp.tasks.BaseOptions(model_asset_path=str(face_model_path))
            landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=2,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
                landmarker_options
            )
            frame_timestamp = [0]
            outline_landmarks = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                361, 288, 397, 365, 379, 378, 400, 152, 176, 149,
                150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
                54, 103, 67, 109,
            ]

            def estimate_head_pose(landmarks, frame_width, frame_height):
                landmark_ids = [1, 152, 33, 263, 61, 291]
                image_points = np.array(
                    [
                        (landmarks[index].x * frame_width, landmarks[index].y * frame_height)
                        for index in landmark_ids
                    ],
                    dtype=np.float64,
                )
                model_points = np.array(
                    [
                        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
                        (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
                        (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
                    ],
                    dtype=np.float64,
                )
                camera_matrix = np.array(
                    [[frame_width, 0, frame_width / 2], [0, frame_width, frame_height / 2], [0, 0, 1]],
                    dtype=np.float64,
                )
                success, rotation_vector, _ = cv2.solvePnP(
                    model_points, image_points, camera_matrix, np.zeros((4, 1)),
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not success:
                    return "Head pose unavailable"
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                pitch, yaw, _ = cv2.RQDecomp3x3(rotation_matrix)[0]
                if yaw < -12:
                    return "Looking left"
                if yaw > 12:
                    return "Looking right"
                if pitch < -12:
                    return "Looking down"
                if pitch > 12:
                    return "Looking up"
                return "Looking forward"

            def process_face_mesh_frame(frame):
                image = frame.to_ndarray(format="bgr24")
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frame_timestamp[0] += 33
                result = face_landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image),
                    frame_timestamp[0],
                )

                if result.face_landmarks:
                    frame_height, frame_width = image.shape[:2]
                    for landmarks in result.face_landmarks:
                        points = np.array(
                            [(int(point.x * frame_width), int(point.y * frame_height)) for point in landmarks],
                            dtype=np.int32,
                        )
                        if mesh_detail == "Full face mesh":
                            for point in points:
                                cv2.circle(image, tuple(point), 1, (0, 255, 0), -1)
                        outline = points[outline_landmarks]
                        cv2.polylines(image, [outline], True, (0, 255, 0), 2)
                        direction = estimate_head_pose(landmarks, frame_width, frame_height)
                        cv2.putText(image, direction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(image, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                return av.VideoFrame.from_ndarray(image, format="bgr24")

            webrtc_streamer(
                key="live-face-mesh",
                video_frame_callback=process_face_mesh_frame,
                media_stream_constraints={"video": True, "audio": False},
            )

# LIVE WEBCAM DETECTION PAGE
elif option == "Live Webcam Detection":

    st.header("Live Webcam Detection")
    st.write(
        "Start your camera to detect and track people and vehicle types in real time. "
        "Tracking IDs are temporary and do not identify people."
    )

    live_confidence = st.slider(
        "Live Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        key="live_confidence",
    )
    live_classes = st.multiselect(
        "Detect in live video",
        options=LIVE_DETECTION_CLASSES,
        default=LIVE_DETECTION_CLASSES,
        key="live_classes",
    )
    live_zoom = st.slider(
        "Camera zoom",
        min_value=1.0,
        max_value=3.0,
        value=1.0,
        step=0.25,
        key="live_zoom",
    )
    live_pan_x = st.slider(
        "Camera pan horizontally",
        min_value=0,
        max_value=100,
        value=50,
        key="live_pan_x",
        disabled=live_zoom == 1.0,
    )
    live_pan_y = st.slider(
        "Camera pan vertically",
        min_value=0,
        max_value=100,
        value=50,
        key="live_pan_y",
        disabled=live_zoom == 1.0,
    )
    roi_enabled = st.checkbox("Detect only inside a selected region", key="roi_enabled")
    roi_x_range = st.slider(
        "Detection region: horizontal range",
        min_value=0,
        max_value=100,
        value=(0, 100),
        key="roi_x_range",
        disabled=not roi_enabled,
    )
    roi_y_range = st.slider(
        "Detection region: vertical range",
        min_value=0,
        max_value=100,
        value=(0, 100),
        key="roi_y_range",
        disabled=not roi_enabled,
    )
    counter_enabled = st.checkbox("Count objects crossing a horizontal line", key="counter_enabled")
    count_line_position = st.slider(
        "Counting line position",
        min_value=10,
        max_value=90,
        value=50,
        key="count_line_position",
        disabled=not counter_enabled,
    )
    blur_faces = st.checkbox("Blur faces in the displayed live video", key="blur_faces")
    known_faces_for_live = load_known_faces()
    recognize_known_faces = st.checkbox(
        "Recognize registered faces when a person alert is triggered",
        key="recognize_known_faces",
        disabled=not known_faces_for_live,
        help="Matches only people enrolled on the Known Face Registration page.",
    )
    if not known_faces_for_live:
        st.caption("Register a consenting person on the Known Face Registration page to enable face recognition.")
    enhance_low_light = st.checkbox(
        "Enhance low-light face recognition",
        value=True,
        key="enhance_low_light",
    )
    show_live_chart = st.checkbox(
        "Show live people and vehicle counter chart",
        value=True,
        key="show_live_chart",
    )
    live_processing_stride = st.select_slider(
        "Live processing speed",
        options=[1, 2, 3],
        value=3,
        help="3 is recommended for a stable webcam. 1 processes every camera frame but can freeze slower laptops.",
        key="live_processing_stride_stable",
    )
    alert_classes = st.multiselect(
        "Show alert when these objects appear",
        options=LIVE_DETECTION_CLASSES,
        key="alert_classes",
    )
    alert_confidence = st.slider(
        "Alert confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.60,
        step=0.05,
        key="alert_confidence",
        disabled=not alert_classes,
    )
    sound_alert = st.checkbox(
        "Play a sound when an alert is detected",
        value=True,
        key="sound_alert",
        disabled=not alert_classes,
    )
    alert_location = st.text_input(
        "Alert location",
        placeholder="Example: Main gate, Building A",
        key="alert_location",
        help="This label is shown on the alert and included in its email.",
    )
    share_gps_location = st.checkbox(
        "Share this browser's GPS location in known-face alerts",
        key="share_gps_location",
    )
    advanced_gps = st.checkbox(
        "Use advanced high-accuracy GPS",
        value=True,
        key="advanced_gps",
        help="Uses fresh device GPS when available. It can take up to 20 seconds.",
        disabled=not share_gps_location,
    )
    if "gps_request_number" not in st.session_state:
        st.session_state["gps_request_number"] = 0
    if share_gps_location and st.button("Request GPS location"):
        st.session_state["gps_request_number"] += 1
    gps_location = st.session_state.get("approved_gps_location", "")
    if share_gps_location:
        try:
            gps_component_key = f"known_face_alert_gps_{st.session_state['gps_request_number']}"
            if advanced_gps:
                geolocation = request_high_accuracy_geolocation(gps_component_key)
            else:
                from streamlit_js_eval import get_geolocation
                geolocation = get_geolocation(component_key=gps_component_key)
            if isinstance(geolocation, str):
                try:
                    geolocation = json.loads(geolocation)
                except json.JSONDecodeError:
                    geolocation = None
            coordinates = geolocation.get("coords", {}) if isinstance(geolocation, dict) else {}
            latitude = coordinates.get("latitude")
            longitude = coordinates.get("longitude")
            if latitude is not None and longitude is not None:
                accuracy = coordinates.get("accuracy")
                accuracy_label = (
                    f" (accuracy ±{float(accuracy):.0f} m)"
                    if accuracy is not None else ""
                )
                gps_location = (
                    f"GPS: {float(latitude):.6f}, {float(longitude):.6f}"
                    f"{accuracy_label}"
                )
                st.session_state["approved_gps_location"] = gps_location
                st.success(f"GPS location ready: {gps_location}")
                if accuracy is not None and float(accuracy) > 50:
                    st.warning(
                        "GPS accuracy is currently low. For a more exact point, use a phone with GPS "
                        "or move to an area with stronger location signals."
                    )
            else:
                error = geolocation.get("error", {}) if isinstance(geolocation, dict) else {}
                if error:
                    st.warning(
                        "GPS could not be read: "
                        f"{error.get('message', 'browser location permission is unavailable')}"
                    )
                elif gps_location:
                    st.success(f"Using saved GPS location: {gps_location}")
                else:
                    st.info("Allow location permission in your browser to share GPS coordinates.")
        except Exception:
            st.warning("Browser GPS is unavailable. You can still enter an alert location manually.")
    alert_location_label = gps_location or alert_location.strip()
    save_alert_snapshots = st.checkbox(
        "Save alert snapshots to the gallery",
        key="save_alert_snapshots",
        disabled=not alert_classes,
    )
    email_alert_configured = email_alert_is_configured()
    send_email_alerts = st.checkbox(
        "Send email alerts with snapshots",
        key="send_email_alerts",
        disabled=not alert_classes or not email_alert_configured,
    )
    if not email_alert_configured:
        st.caption("Configure email settings in the local .env file to enable email alerts.")
    if turn_server_is_configured():
        st.caption("Webcam connection: STUN + TURN enabled.")
    else:
        st.caption("Webcam connection: STUN enabled. Add TURN settings to .env if your network blocks the camera stream.")
    live_event_log = get_live_event_log()
    alert_snapshot_gallery = get_alert_snapshot_gallery()
    if "webcam_connection_generation" not in st.session_state:
        st.session_state["webcam_connection_generation"] = 0
    if st.button("Restart Webcam Connection"):
        st.session_state["webcam_connection_generation"] += 1
        st.rerun()
    if st.button("Clear Live Event Log"):
        live_event_log.clear()
    if st.button("Clear Alert Snapshot Gallery"):
        alert_snapshot_gallery.clear()

    try:
        import av
        from streamlit_webrtc import webrtc_streamer
    except ImportError:
        st.warning(
            "Install the updated requirements to enable the live webcam feature: "
            "python -m pip install -r requirements.txt"
        )
    else:
        if not live_classes:
            st.info("Select at least one object type to start live detection.")
        else:
            live_class_ids = class_ids_for_names(model.names, live_classes)
            live_tracking_model = YOLO(model_paths[model_label])
            live_face_detector = load_face_detector()
            previous_centers = {}
            counted_in = set()
            counted_out = set()
            live_counts = {"in": 0, "out": 0}
            last_frame_time = [time.perf_counter()]
            last_sound_time = [0.0]
            last_snapshot_time = [0.0]
            last_email_time = [0.0]
            people_history = []
            vehicle_history = []
            seen_event_track_ids = set()
            recognition_frame_number = [0]
            recent_known_matches = []
            live_face_confirmation_counts = Counter()
            known_face_log_times = {}
            known_face_worker = KnownFaceRecognitionWorker()
            last_recognition_request_time = [0.0]
            live_frame_number = [0]
            last_live_display = [None]
            last_processed_frame_time = [time.perf_counter()]

            def draw_live_chart(image, people_counts, vehicle_counts):
                chart_width = min(240, image.shape[1] - 20)
                chart_height = min(110, image.shape[0] - 20)
                chart_x = image.shape[1] - chart_width - 10
                chart_y = image.shape[0] - chart_height - 10
                cv2.rectangle(
                    image,
                    (chart_x, chart_y),
                    (chart_x + chart_width, chart_y + chart_height),
                    (20, 20, 20),
                    -1,
                )
                cv2.rectangle(
                    image,
                    (chart_x, chart_y),
                    (chart_x + chart_width, chart_y + chart_height),
                    (220, 220, 220),
                    1,
                )
                maximum_count = max(max(people_counts, default=0), max(vehicle_counts, default=0), 1)

                def chart_points(values):
                    if len(values) == 1:
                        return np.array([[chart_x, chart_y + chart_height]], dtype=np.int32)
                    return np.array(
                        [
                            (
                                chart_x + int(index * (chart_width - 1) / (len(values) - 1)),
                                chart_y + chart_height - int(value * (chart_height - 20) / maximum_count),
                            )
                            for index, value in enumerate(values)
                        ],
                        dtype=np.int32,
                    )

                if people_counts:
                    cv2.polylines(image, [chart_points(people_counts)], False, (0, 255, 0), 2)
                if vehicle_counts:
                    cv2.polylines(image, [chart_points(vehicle_counts)], False, (0, 165, 255), 2)
                cv2.putText(image, "People", (chart_x + 6, chart_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.putText(image, "Vehicles", (chart_x + 70, chart_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

            def process_live_frame(frame):
                image = frame.to_ndarray(format="bgr24")
                frame_height, frame_width = image.shape[:2]
                crop_width = max(1, int(frame_width / live_zoom))
                crop_height = max(1, int(frame_height / live_zoom))
                max_x = frame_width - crop_width
                max_y = frame_height - crop_height
                crop_x = round(max_x * live_pan_x / 100)
                crop_y = round(max_y * live_pan_y / 100)
                image = image[
                    crop_y:crop_y + crop_height,
                    crop_x:crop_x + crop_width,
                ]
                live_frame_number[0] += 1
                if (
                    last_live_display[0] is not None
                    and live_frame_number[0] % live_processing_stride != 0
                ):
                    return av.VideoFrame.from_ndarray(last_live_display[0], format="bgr24")
                image_height, image_width = image.shape[:2]
                if roi_enabled:
                    roi_x1 = min(round(image_width * roi_x_range[0] / 100), image_width - 1)
                    roi_x2 = max(roi_x1 + 1, round(image_width * roi_x_range[1] / 100))
                    roi_y1 = min(round(image_height * roi_y_range[0] / 100), image_height - 1)
                    roi_y2 = max(roi_y1 + 1, round(image_height * roi_y_range[1] / 100))
                else:
                    roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, image_width, image_height
                detection_image = image[roi_y1:roi_y2, roi_x1:roi_x2]
                results = live_tracking_model.track(
                    detection_image,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=live_confidence,
                    classes=live_class_ids,
                    verbose=False,
                )
                annotated_image = results[0].plot()
                display_image = image.copy()
                display_image[roi_y1:roi_y2, roi_x1:roi_x2] = annotated_image

                active_ids = []
                alerts = []
                people_count = 0
                vehicle_count = 0
                person_detected = False
                recognized_names = []
                line_y = round((roi_y2 - roi_y1) * count_line_position / 100)
                for box in results[0].boxes:
                    class_name = model.names[int(box.cls[0])]
                    confidence = float(box.conf[0])
                    if class_name == "person":
                        people_count += 1
                        person_detected = True
                    elif class_name in VEHICLE_CLASSES:
                        vehicle_count += 1
                    if class_name in alert_classes and confidence >= alert_confidence:
                        alerts.append(class_name)
                    if box.id is not None:
                        track_id = int(box.id[0])
                        active_ids.append(track_id)
                        if track_id not in seen_event_track_ids:
                            live_event_log.add({
                                "Time": datetime.now().strftime("%H:%M:%S"),
                                "Object": class_name,
                                "Track ID": track_id,
                                "Confidence": f"{confidence:.0%}",
                            })
                            seen_event_track_ids.add(track_id)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        center_y = (y1 + y2) / 2
                        previous_y = previous_centers.get(track_id)
                        if counter_enabled and previous_y is not None:
                            if previous_y < line_y <= center_y and track_id not in counted_in:
                                counted_in.add(track_id)
                                live_counts["in"] += 1
                            elif previous_y > line_y >= center_y and track_id not in counted_out:
                                counted_out.add(track_id)
                                live_counts["out"] += 1
                        previous_centers[track_id] = center_y

                people_history.append(people_count)
                vehicle_history.append(vehicle_count)
                del people_history[:-90]
                del vehicle_history[:-90]

                if counter_enabled:
                    cv2.line(
                        display_image,
                        (roi_x1, roi_y1 + line_y),
                        (roi_x2, roi_y1 + line_y),
                        (0, 255, 255),
                        2,
                    )

                recognition_frame_number[0] += 1
                if blur_faces or (
                    recognize_known_faces
                    and person_detected
                    and bool(alerts)
                    and recognition_frame_number[0] % 3 == 0
                ):
                    face_detection_image = (
                        enhance_low_light_image(display_image)
                        if enhance_low_light else display_image
                    )
                    gray_image = cv2.cvtColor(face_detection_image, cv2.COLOR_BGR2GRAY)
                    for x, y, width, height in live_face_detector.detectMultiScale(
                        gray_image,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(40, 40),
                    ):
                        face_region = face_detection_image[y:y + height, x:x + width]
                        match = None
                        recognition_time = time.perf_counter()
                        if recognize_known_faces and person_detected and alerts and recognition_time - last_recognition_request_time[0] >= 5:
                            known_face_worker.request(
                                face_region,
                                known_faces_for_live,
                                (x + width / 2, y + height / 2),
                            )
                            last_recognition_request_time[0] = recognition_time
                        if recognize_known_faces and person_detected and alerts:
                            latest_match = known_face_worker.get_latest_match()
                            if latest_match:
                                name, similarity, match_x, match_y = latest_match
                                recent_known_matches[:] = [
                                    entry for entry in recent_known_matches
                                    if entry[0] != name
                                ]
                                recent_known_matches.append((name, similarity, match_x, match_y))
                            nearby_matches = [
                                entry for entry in recent_known_matches
                                if abs(entry[2] - (x + width / 2)) < max(width, 80)
                                and abs(entry[3] - (y + height / 2)) < max(height, 80)
                            ]
                            if nearby_matches:
                                name, similarity, _, _ = nearby_matches[-1]
                                match = (name, similarity)
                        if match:
                            name, similarity = match
                            live_face_confirmation_counts[name] += 1
                            confirmed = live_face_confirmation_counts[name] >= 2
                            if confirmed:
                                recognized_names.append(name)
                            cv2.rectangle(display_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                            cv2.putText(
                                display_image,
                                f"{name} ({similarity:.0%})" if confirmed else f"Confirming {name}...",
                                (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 255, 0),
                                2,
                            )
                            if alert_location_label:
                                cv2.putText(
                                    display_image,
                                    alert_location_label,
                                    (x, min(display_image.shape[0] - 10, y + height + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45,
                                    (0, 255, 255),
                                    1,
                                )
                            if recognition_time - known_face_log_times.get(name, 0.0) >= 30:
                                saved_location = alert_location_label or "Not set"
                                live_event_log.add({
                                    "Time": datetime.now().strftime("%H:%M:%S"),
                                    "Object": f"Known face: {name}",
                                    "Track ID": "-",
                                    "Confidence": f"{similarity:.0%}",
                                    "Location": saved_location,
                                })
                                save_location_alert(name, f"{similarity:.0%}", saved_location)
                                known_face_log_times[name] = recognition_time
                        if blur_faces:
                            original_face_region = display_image[y:y + height, x:x + width]
                            display_image[y:y + height, x:x + width] = cv2.GaussianBlur(
                                original_face_region,
                                (31, 31),
                                0,
                            )

                if roi_enabled:
                    cv2.rectangle(
                        display_image,
                        (roi_x1, roi_y1),
                        (roi_x2, roi_y2),
                        (255, 255, 0),
                        2,
                    )
                current_time = time.perf_counter()
                fps = 1 / max(current_time - last_frame_time[0], 0.000001)
                last_frame_time[0] = current_time
                overlay_x = max(15, display_image.shape[1] - 410)
                status_text = (
                    f"FPS: {fps:.1f} | Active: {len(results[0].boxes)} | "
                    f"Track IDs: {', '.join(map(str, active_ids)) or 'None'}"
                )
                cv2.putText(
                    display_image,
                    status_text,
                    (overlay_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                if counter_enabled:
                    cv2.putText(
                        display_image,
                        f"Crossed in: {live_counts['in']} | Crossed out: {live_counts['out']}",
                        (overlay_x, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )
                if alerts:
                    cv2.putText(
                        display_image,
                        f"ALERT: {', '.join(sorted(set(alerts)))} detected",
                        (overlay_x, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )
                    if alert_location_label:
                        cv2.putText(
                            display_image,
                            f"Location: {alert_location_label}",
                            (overlay_x, 105),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 255),
                            2,
                        )
                    if sound_alert and current_time - last_sound_time[0] >= 3:
                        try:
                            import winsound

                            winsound.Beep(1000, 250)
                        except RuntimeError:
                            pass
                        last_sound_time[0] = current_time
                    if save_alert_snapshots and current_time - last_snapshot_time[0] >= 5:
                        encoded, snapshot = cv2.imencode(".png", display_image)
                        if encoded:
                            alert_snapshot_gallery.add({
                                "Time": datetime.now().strftime("%H:%M:%S"),
                                "Objects": ", ".join(sorted(set(alerts))),
                                "Image": snapshot.tobytes(),
                            })
                            last_snapshot_time[0] = current_time
                    email_alert_ready = bool(alerts) and (
                        not recognize_known_faces or bool(recognized_names)
                    )
                    if send_email_alerts and email_alert_ready and current_time - last_email_time[0] >= 60:
                        encoded, email_snapshot = cv2.imencode(".png", display_image)
                        if encoded:
                            alert_names = sorted(
                                set(alerts + [f"Known face: {name}" for name in recognized_names])
                            )
                            send_email_alert_in_background(
                                alert_names,
                                email_snapshot.tobytes(),
                                live_event_log,
                                alert_location_label,
                            )
                            last_email_time[0] = current_time
                if show_live_chart:
                    draw_live_chart(display_image, people_history, vehicle_history)
                last_live_display[0] = display_image.copy()
                last_processed_frame_time[0] = time.perf_counter()
                return av.VideoFrame.from_ndarray(display_image, format="bgr24")

            webcam_context = webrtc_streamer(
                key=(
                    f"live-object-detection-{alert_location_label}-"
                    f"{st.session_state['webcam_connection_generation']}"
                ),
                video_frame_callback=process_live_frame,
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration=get_webrtc_configuration(),
                async_processing=True,
            )

            @st.fragment(run_every=5)
            def show_webcam_connection_health():
                st.subheader("Webcam Connection Health")
                if webcam_context.state.playing:
                    seconds_since_frame = time.perf_counter() - last_processed_frame_time[0]
                    if seconds_since_frame > 12:
                        st.warning(
                            "Camera is connected but frame processing has stopped. "
                            "Use Restart Webcam Connection above."
                        )
                    else:
                        st.success("Camera connected and processing frames.")
                else:
                    st.info(
                        "Waiting for camera connection. If you just pressed START, allow a few seconds for WebRTC. "
                        "If it does not connect, use Restart Webcam Connection."
                    )

            show_webcam_connection_health()

            @st.fragment(run_every=5)
            def show_live_event_log():
                st.subheader("Live Event Log")
                events = live_event_log.snapshot()
                if events:
                    event_table = pd.DataFrame(events)
                    if "Track ID" in event_table:
                        event_table["Track ID"] = event_table["Track ID"].fillna("-").astype(str)
                    st.dataframe(
                        event_table,
                        use_container_width=True,
                        hide_index=True,
                        height=260,
                    )
                    st.download_button(
                        "Download Live Event Log (CSV)",
                        data=event_table.to_csv(index=False).encode("utf-8"),
                        file_name="live_event_log.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Start the webcam to record detected tracked objects.")

            show_live_event_log()

            @st.fragment(run_every=5)
            def show_alert_snapshot_gallery():
                st.subheader("Alert Snapshot Gallery")
                snapshots = alert_snapshot_gallery.snapshot()
                if snapshots:
                    columns = st.columns(3)
                    for index, snapshot in enumerate(snapshots):
                        with columns[index % 3]:
                            st.image(
                                snapshot["Image"],
                                caption=f"{snapshot['Time']} - {snapshot['Objects']}",
                                use_container_width=True,
                            )
                            st.download_button(
                                "Download Snapshot",
                                data=snapshot["Image"],
                                file_name=f"alert_snapshot_{snapshot['Time'].replace(':', '-')}.png",
                                mime="image/png",
                                key=f"alert_snapshot_{index}_{snapshot['Time']}",
                            )
                else:
                    st.info("Enable alert snapshots and detect a selected object to build the gallery.")

            show_alert_snapshot_gallery()

# OBJECT TRACKING PAGE
elif option == "Object Tracking":

    st.header("🎥 Video Object Tracking")

    st.write(
        "Upload a video to detect and track objects "
        "using YOLO and ByteTrack."
    )

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.subheader("Original Video")
        st.video(uploaded_video)

        tracking_confidence = st.slider(
            "Tracking Confidence",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05
        )
        video_alert_classes = st.multiselect(
            "Alert when these objects appear in the video",
            options=list(model.names.values()),
            default=["person"],
            key="video_alert_classes",
        )
        video_alert_confidence = st.slider(
            "Video alert confidence",
            min_value=0.10,
            max_value=0.90,
            value=0.60,
            step=0.05,
            key="video_alert_confidence",
            disabled=not video_alert_classes,
        )
        video_sound_alert = st.checkbox(
            "Play one sound when a video alert is found",
            key="video_sound_alert",
            disabled=not video_alert_classes,
        )
        known_faces_for_video = load_known_faces()
        recognize_video_known_faces = st.checkbox(
            "Recognize registered faces in this video",
            key="video_known_face_recognition",
            disabled=not known_faces_for_video,
            help="Matches only people enrolled on the Known Face Registration page.",
        )
        video_face_check_interval = st.select_slider(
            "Face matching frequency (frames)",
            options=[15, 30, 60],
            value=30,
            key="video_face_check_interval",
            disabled=not recognize_video_known_faces,
            help="Checking fewer frames keeps video processing faster. 30 is recommended.",
        )
        if not known_faces_for_video:
            st.caption("Register a consenting person on the Known Face Registration page to enable face matching.")

        if st.button("Start Object Tracking"):

            input_path = None
            output_path = None
            browser_path = None
            video = None
            writer = None

            try:
                # Save uploaded video temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(uploaded_video.name)[1]
                ) as temp_input:

                    temp_input.write(uploaded_video.getvalue())
                    input_path = temp_input.name

                # Open uploaded video
                video = cv2.VideoCapture(input_path)

                if not video.isOpened():
                    st.error("Unable to open the uploaded video.")

                else:
                    fps = video.get(cv2.CAP_PROP_FPS)

                    if fps <= 0:
                        fps = 25.0

                    width = int(
                        video.get(cv2.CAP_PROP_FRAME_WIDTH)
                    )

                    height = int(
                        video.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    )

                    if width <= 0 or height <= 0:
                        st.error("Unable to read video dimensions.")

                    else:
                        # Create temporary output video
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp4"
                        ) as temp_output:
                            output_path = temp_output.name

                        writer = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (width, height)
                        )

                        if not writer.isOpened():
                            st.error("Unable to create output video.")

                        else:
                            progress = st.progress(0)
                            status = st.empty()

                            total_frames = int(
                                video.get(cv2.CAP_PROP_FRAME_COUNT)
                            )

                            processed_frames = 0
                            unique_track_ids = set()
                            tracked_category_counts = Counter()
                            tracking_snapshot = None
                            video_alert_events = []
                            video_alert_snapshot = None
                            video_alarm_played = False
                            video_known_face_events = []
                            video_known_face_snapshot = None

                            # Start tracking with a fresh tracker state
                            tracking_model = YOLO(model_paths[model_label])

                            while True:

                                success, frame = video.read()

                                if not success:
                                    break

                                results = tracking_model.track(
                                    frame,
                                    persist=True,
                                    tracker="bytetrack.yaml",
                                    conf=tracking_confidence,
                                    verbose=False
                                )

                                annotated_frame = results[0].plot()

                                # Face embeddings are expensive, so only inspect frames with a
                                # detected person at the chosen interval.
                                person_in_frame = any(
                                    model.names[int(box.cls[0])] == "person"
                                    for box in results[0].boxes
                                )
                                if (
                                    recognize_video_known_faces
                                    and person_in_frame
                                    and processed_frames % video_face_check_interval == 0
                                ):
                                    annotated_frame, frame_face_matches = recognize_known_faces_in_frame(
                                        annotated_frame,
                                        known_faces_for_video,
                                    )
                                    for face_match in frame_face_matches:
                                        event = {"Frame": processed_frames + 1, **face_match}
                                        video_known_face_events.append(event)
                                    if frame_face_matches and video_known_face_snapshot is None:
                                        encoded, snapshot = cv2.imencode(".png", annotated_frame)
                                        if encoded:
                                            video_known_face_snapshot = snapshot.tobytes()

                                if tracking_snapshot is None:
                                    encoded, snapshot = cv2.imencode(".png", annotated_frame)
                                    if encoded:
                                        tracking_snapshot = snapshot.tobytes()

                                for box in results[0].boxes:
                                    class_id = int(box.cls[0])
                                    class_name = model.names[class_id]
                                    confidence = float(box.conf[0])
                                    tracked_category_counts[class_name] += 1
                                    if box.id is not None:
                                        unique_track_ids.add(int(box.id[0]))
                                    if class_name in video_alert_classes and confidence >= video_alert_confidence:
                                        video_alert_events.append({
                                            "Frame": processed_frames + 1,
                                            "Object": class_name,
                                            "Confidence": f"{confidence:.0%}",
                                        })

                                frame_alerts = sorted({event["Object"] for event in video_alert_events if event["Frame"] == processed_frames + 1})
                                if frame_alerts:
                                    cv2.putText(
                                        annotated_frame,
                                        f"ALERT: {', '.join(frame_alerts)} detected",
                                        (15, 35),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 0, 255),
                                        2,
                                    )
                                    if video_alert_snapshot is None:
                                        encoded, snapshot = cv2.imencode(".png", annotated_frame)
                                        if encoded:
                                            video_alert_snapshot = snapshot.tobytes()
                                    if video_sound_alert and not video_alarm_played:
                                        try:
                                            import winsound
                                            winsound.Beep(1000, 350)
                                        except RuntimeError:
                                            pass
                                        video_alarm_played = True

                                writer.write(annotated_frame)

                                processed_frames += 1

                                if total_frames > 0:
                                    progress.progress(
                                        min(
                                            processed_frames / total_frames,
                                            1.0
                                        )
                                    )

                                status.write(
                                    f"Processed frames: {processed_frames}"
                                )

                            writer.release()
                            writer = None

                            if processed_frames == 0:
                                st.warning(
                                    "No video frames were processed."
                                )

                            else:
                                st.info(
                                    "Preparing video for browser playback..."
                                )

                                # Convert output to browser-compatible MP4
                                with tempfile.NamedTemporaryFile(
                                    delete=False,
                                    suffix=".mp4"
                                ) as temp_browser:
                                    browser_path = temp_browser.name

                                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

                                subprocess.run(
                                    [
                                        ffmpeg,
                                        "-y",
                                        "-i", output_path,
                                        "-c:v", "libx264",
                                        "-pix_fmt", "yuv420p",
                                        "-movflags", "+faststart",
                                        browser_path
                                    ],
                                    check=True,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL
                                )

                                with open(browser_path, "rb") as file:
                                    video_bytes = file.read()

                                st.success("Object Tracking Completed!")

                                summary_one, summary_two, summary_three = st.columns(3)
                                summary_one.metric("Frames Processed", processed_frames)
                                summary_two.metric("Unique Tracked Objects", len(unique_track_ids))
                                summary_three.metric("Object Categories", len(tracked_category_counts))
                                st.metric("Video Alert Events", len(video_alert_events))

                                if recognize_video_known_faces:
                                    st.metric("Registered Face Matches", len(video_known_face_events))
                                    if video_known_face_events:
                                        st.subheader("Registered Face Matches")
                                        st.dataframe(
                                            pd.DataFrame(video_known_face_events),
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                                        if video_known_face_snapshot:
                                            st.download_button(
                                                "Download First Known Face Match",
                                                data=video_known_face_snapshot,
                                                file_name="known_face_video_match.png",
                                                mime="image/png",
                                            )
                                    else:
                                        st.info("No registered face was matched in this video.")

                                if video_alert_events:
                                    st.warning(f"Alert found: {', '.join(sorted(set(event['Object'] for event in video_alert_events)))}")
                                    st.dataframe(pd.DataFrame(video_alert_events), use_container_width=True, hide_index=True)
                                    if video_alert_snapshot:
                                        st.download_button(
                                            "Download First Alert Snapshot",
                                            data=video_alert_snapshot,
                                            file_name="video_alert_snapshot.png",
                                            mime="image/png",
                                        )

                                if tracked_category_counts:
                                    tracking_counts = pd.DataFrame(
                                        tracked_category_counts.items(),
                                        columns=["Object", "Frame Detections"],
                                    ).sort_values("Frame Detections", ascending=False)
                                    st.subheader("Tracking Summary")
                                    st.dataframe(
                                        tracking_counts,
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                                st.subheader("Tracked Video")

                                st.video(video_bytes)

                                if tracking_snapshot is not None:
                                    st.download_button(
                                        label="Download Tracking Screenshot",
                                        data=tracking_snapshot,
                                        file_name="tracking_screenshot.png",
                                        mime="image/png",
                                    )

                                st.download_button(
                                    label="Download Tracked Video",
                                    data=video_bytes,
                                    file_name="tracked_video.mp4",
                                    mime="video/mp4"
                                )

            except Exception as error:
                st.error(f"Processing error: {error}")

            finally:
                if video is not None:
                    video.release()

                if writer is not None:
                    writer.release()

                # Remove temporary files
                for path in [
                    input_path,
                    output_path,
                    browser_path
                ]:
                    if path and os.path.exists(path):
                        os.remove(path)
