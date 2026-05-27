@echo off
chcp 65001 >nul
cd /d "C:\Users\ASUS\Desktop\ai-build-ai\action_c\demo\edge_video_preprocess"
python gen_cifar_video.py --frames 200 --obj-size 14 --motion-len 12 --static-len 30 %*
pause
