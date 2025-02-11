import sys
import subprocess
import os

if len(sys.argv) < 2:
    print("Usage: python video_downloader.py <YouTube_URL>")
    sys.exit(1)

youtube_url = sys.argv[1]

# Find the next available filename (cameraX.mp4)
counter = 1
while os.path.exists(f"camera{counter}.mp4"):
    counter += 1

output_filename = f"camera{counter}.mp4"

# Download the video in MP4 format (video and audio merged)
try:
    # This command ensures the download of both video and audio streams, and forces MP4 format for output
    subprocess.run(
        ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4", "-o", output_filename, youtube_url],
        check=True
    )
    print(f"Video downloaded successfully as {output_filename}")
except subprocess.CalledProcessError as e:
    print(f"Error downloading video: {e}")
