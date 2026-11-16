import os
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "mission5_artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for name in ("message_distilbert", "attachment_distilbert"):
    source = os.path.join(BASE_DIR, "models", name)
    if os.path.isdir(source):
        archive = shutil.make_archive(os.path.join(OUTPUT_DIR, name), "zip", source)
        print(f"Created: {archive}")

reports = os.path.join(BASE_DIR, "reports")
report_archive = shutil.make_archive(os.path.join(OUTPUT_DIR, "mission5_reports"), "zip", reports)
print(f"Created: {report_archive}")