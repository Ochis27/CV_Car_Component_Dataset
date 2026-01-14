import os

FOLDER_PATH = "/Users/ochispaul-catalin/Desktop/CV_Project/Compoent_Images"

files = os.listdir(FOLDER_PATH)
image_extensions = (".jpg", ".jpeg", ".png")
image_files = [f for f in files if f.lower().endswith(image_extensions)]
image_files.sort()

for index, filename in enumerate(image_files, start=1):
    ext = os.path.splitext(filename)[1].lower()
    old_path = os.path.join(FOLDER_PATH, filename)
    new_path = os.path.join(FOLDER_PATH, f"{index}{ext}")
    os.rename(old_path, new_path)

print(f"Renamed {len(image_files)} images successfully.")
