ffmpeg -i data/videos/0.mp4 -vf "select=eq(n\,0)" -q:v 3 data/images/0.jpg
