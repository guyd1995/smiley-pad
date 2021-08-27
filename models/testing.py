import torch
from time import sleep
import cv2
from train import preprocess_img
from PIL import Image, ImageFont, ImageDraw
import json
import numpy as np

model_path = "face_model.ptl"
idx2emoji_path = "idx2emoji.json"

with open(idx2emoji_path) as f:
    idx2emoji = json.load(f)

model = torch.jit.load(model_path).eval()
font = ImageFont.truetype("Seguiemj.ttf", 32)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 30)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


while True:
    ret, frame = cap.read()
    rgb = frame #cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cv2.imshow('frame', rgb)
    key = cv2.waitKey(10) & 0xFF
    if key == ord(" "):
        out = model(preprocess_img(rgb, resize=False).unsqueeze(0)).squeeze(0).detach().numpy()
        res = np.argmax(out)
        cur_emoji = idx2emoji[res]
        if cur_emoji is None:
            cur_emoji = "NEUTRAL"
        pil_img = Image.fromarray(rgb)
        pil_draw = ImageDraw.Draw(pil_img)
        pil_draw.text((40, 80), cur_emoji, (255,255,255), font=font)
        rgb = np.array(pil_img)
        cv2.imshow('frame', rgb)
        cv2.waitKey(1000)
    if key == ord("q"): 
        break 

cap.release()
cv2.destroyAllWindows()