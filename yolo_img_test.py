import argparse
import time
import glob
import os
import cv2
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="",
                    help="path to (optional) input images directory")
parser.add_argument("-o", "--output", type=str, default="",
                    help="path to (optional) output image directory")
parser.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
parser.add_argument("-t", "--threshold", type=float, default=0.4,
                    help="threshold for non maxima supression")

args = vars(parser.parse_args())

CONFIDENCE_THRESHOLD = args["confidence"]
NMS_THRESHOLD = args["threshold"]
path_to_folder = args["input"]

weights = glob.glob("models/*.weights")[0]
labels = glob.glob("models/*.txt")[0]
cfg = glob.glob("models/*.cfg")[0]

print("You are now using {} weights ,{} configs and {} labels.".format(weights, cfg, labels))
start = datetime.now()

lbls = list()
with open(labels, "r") as f:
    lbls = [c.strip() for c in f.readlines()]

COLORS = np.random.randint(0, 255, size=(len(lbls), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer = net.getLayerNames()
layer = [layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def detect(imgpath, nn):
    image = cv2.imread(imgpath)
    img = image.copy()
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    nn.setInput(blob)
    start_time = time.time()
    layer_outs = nn.forward(layer)
    end_time = time.time()
    coords = []
    crp_imgs = []
    boxes = list()
    confidences = list()
    class_ids = list()

    for output in layer_outs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # color = [int(c) for c in COLORS[class_ids[i]]]
            crop_img = img[y:y + h, x:x + w]
            crp_imgs.append(crop_img)
            coords.append({'x1': x, 'x2': x + w, 'y1': y, 'y2': y + h})
            # cv2.imshow("cropped", crop_img)
            # cv2.waitKey(400)
            # cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            # text = "{}: {:.4f}".format(lbls[class_ids[i]], confidences[i])
            # cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # label = "Inference Time: {:.2f} ms".format(end_time - start_time)
            # cv2.putText(image, label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # cv2.imshow("image", image)
    # if args["output"] != "":
    #     cv2.imwrite(args["output"], image)
    # cv2.waitKey(500)
    return coords,crp_imgs

# print(bounding_boxes)

current_dir = os.getcwd()
if os.path.isdir(os.path.join(current_dir,path_to_folder)):
    img_list = [itm for itm in os.listdir(os.path.join(current_dir,path_to_folder)) if itm[-4:] in (".jpg",".jpeg",".png") ]
    for img_path in img_list:
        bounding_boxes,crp_imgs = detect(os.path.join(current_dir,args['input'],img_path), net)
        count = 0
        for crp_img in crp_imgs:
            count += 1
            if not os.path.isdir(os.path.join(current_dir,args['input'],img_path[:-4])):
                os.makedirs(os.path.join(current_dir,args['input'],img_path[:-4]))
            cv2.imwrite(os.path.join(current_dir,args['input'],img_path[:-4],img_path[:-4]+f"_cropped_{count:04d}"+img_path[-4:]),crp_img)

end = datetime.now()

print(f"Total time has taken to process is {end - start} for count of {count} cropped images!")