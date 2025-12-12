# extract_archives.py
# Script to extract multiple .tar.gz image archives into a single folder

import tarfile
import os

# -------------------------------------------------------
# Paths Configuration
# -------------------------------------------------------

# Directory where archives are located
ARCHIVES_DIR = "F:/gdtnetpp_project"  # images_001.tar.gz … images_012.tar.gz

# Output directory for extracted images
OUT_DIR = os.path.join(ARCHIVES_DIR, "images")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Extract Archives
# -------------------------------------------------------

# Loop through archives images_001.tar.gz … images_012.tar.gz
for i in range(1, 13):
    archive_name = f"images_{i:03}.tar.gz"
    archive_path = os.path.join(ARCHIVES_DIR, archive_name)
    
    if not os.path.exists(archive_path):
        print(f"⚠️ Archive not found: {archive_path}")
        continue

    print(f"📦 Extracting {archive_path} …")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=OUT_DIR)

print("All archives extracted successfully!")
