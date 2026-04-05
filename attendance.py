import csv
import os
from datetime import datetime

from config import ATTENDANCE_DIR


class AttendanceLogger:

    CSV_HEADERS = ["Name", "Date", "Time", "Status"]

    def __init__(self):
        os.makedirs(ATTENDANCE_DIR, exist_ok=True)
        self._filepath = self._today_filepath()
        self._marked: set[str] = set()
        self._init_file()


    @staticmethod
    def _today_filepath() -> str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.csv")

    def _init_file(self):
        if not os.path.exists(self._filepath):
            with open(self._filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)
        else:
            with open(self._filepath, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Name", "").strip()
                    if name:
                        self._marked.add(name)


    def mark(self, name: str) -> bool:
        name = name.strip()
        if not name or name == "Unknown" or name in self._marked:
            return False

        now = datetime.now()
        with open(self._filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                name,
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                "Present",
            ])
        self._marked.add(name)
        return True

    def count(self) -> int:
        return len(self._marked)

    def marked_names(self) -> list[str]:
        return sorted(self._marked)

    @property
    def filepath(self) -> str:
        return self._filepath