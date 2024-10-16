import torch
import cv2
import pathlib
from pathlib import Path

pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')

# Class labels
class_label = {
    0: "anorganik",
    1: "beracun",
    2: "kertas",
    3: "organik",
    4: "residu"
}


# def draw_bounding_boxes(pred_tensor, result):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     fontScale = 0.7
#     size_of_tensor = list(pred_tensor.size())
#     rows = size_of_tensor[0]
#     for i in range(0, rows):
#         cv2.rectangle(result, (int(pred_tensor[i,0].item()), int(pred_tensor[i,1].item())), 
#         (int(pred_tensor[i,2].item()), int(pred_tensor[i,3].item())), (0, 0, 255), 2)

#         text = class_label[int(pred_tensor[i,5].item())] +" " + str(round(pred_tensor[i,4].item(), 2))

#         image = cv2.putText(result, text, (int(pred_tensor[i,0].item())+5, int(pred_tensor[i,1].item())), 
#         font, fontScale, (0, 0, 255), 2)
        
#     return result

def draw_bounding_boxes(pred_tensor, result):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    size_of_tensor = list(pred_tensor.size())
    rows = size_of_tensor[0]
    for i in range(0, rows):
        cv2.rectangle(result, (int(pred_tensor[i,0].item()), int(pred_tensor[i,1].item())), 
        (int(pred_tensor[i,2].item()), int(pred_tensor[i,3].item())), (0, 0, 255), 2)

        text = class_label[int(pred_tensor[i,5].item())] +" " + str(round(pred_tensor[i,4].item(), 2))
        image = cv2.putText(result, text, (int(pred_tensor[i,0].item())-30, int(pred_tensor[i,1].item())), font, fontScale, (0, 0, 255), 2)
        
    return result
# Images
img = cv2.imread("test/crowd_mask38.jpg")
#im = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

print(results.xyxy[0])  # im predictions (tensor)
# print("Shape of tensor:", results.xyxy[0].size())
# print(results.xyxy[0][1, 1].item())
res = draw_bounding_boxes(results.xyxy[0], img)

cv2.imshow("", res)
cv2.waitKey(0)
print(results.pandas().xyxy[0])  # im predictions (pandas)