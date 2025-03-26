import cv2
import numpy as np
import onnxruntime as ort
from class_names import class_names


def preprocess_image(image_path, input_size=(640, 640)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image: {image_path}")
    orig_size = img.shape[:2]
    print(f"Original image size: {orig_size}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
    print(f"Resized image shape: {img_resized.shape}")
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)
    print(f"Input tensor shape: {img_input.shape}")
    return img_input, img, orig_size

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def clamp_box(box, orig_size):
    x_min, y_min, x_max, y_max = box
    x_min = max(0, min(int(x_min), orig_size[1]))
    y_min = max(0, min(int(y_min), orig_size[0]))
    x_max = max(0, min(int(x_max), orig_size[1]))
    y_max = max(0, min(int(y_max), orig_size[0]))
    return [x_min, y_min, x_max, y_max]

def decode_predictions(boxes_data, img_size=640):
    n1 = 80 * 80  # 6400
    n2 = 40 * 40  # 1600
    n3 = 20 * 20  # 400

    # Scale 1: stride=8
    stride1 = 8
    grid_size1 = 80
    pred1 = boxes_data[:n1]  # (6400, 49)
    grid_x, grid_y = np.meshgrid(np.arange(grid_size1), np.arange(grid_size1))
    grid1 = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)
    pred1_xy = sigmoid(pred1[:, :2]) * 2 - 0.5
    pred1_wh = (sigmoid(pred1[:, 2:4]) * 2) ** 2
    cx1 = (grid1[:, 0] + pred1_xy[:, 0]) * stride1
    cy1 = (grid1[:, 1] + pred1_xy[:, 1]) * stride1
    w1 = pred1_wh[:, 0] * stride1
    h1 = pred1_wh[:, 1] * stride1
    x1_1 = cx1 - w1 / 2
    y1_1 = cy1 - h1 / 2
    x2_1 = cx1 + w1 / 2
    y2_1 = cy1 + h1 / 2
    boxes1 = np.stack([x1_1, y1_1, x2_1, y2_1], axis=1)
    class_logits1 = pred1[:, 4:]

    # Scale 2: stride=16
    stride2 = 16
    grid_size2 = 40
    pred2 = boxes_data[n1:n1 + n2]  # (1600, 49)
    grid_x, grid_y = np.meshgrid(np.arange(grid_size2), np.arange(grid_size2))
    grid2 = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)
    pred2_xy = sigmoid(pred2[:, :2]) * 2 - 0.5
    pred2_wh = (sigmoid(pred2[:, 2:4]) * 2) ** 2
    cx2 = (grid2[:, 0] + pred2_xy[:, 0]) * stride2
    cy2 = (grid2[:, 1] + pred2_xy[:, 1]) * stride2
    w2 = pred2_wh[:, 0] * stride2
    h2 = pred2_wh[:, 1] * stride2
    x1_2 = cx2 - w2 / 2
    y1_2 = cy2 - h2 / 2
    x2_2 = cx2 + w2 / 2
    y2_2 = cy2 + h2 / 2
    boxes2 = np.stack([x1_2, y1_2, x2_2, y2_2], axis=1)
    class_logits2 = pred2[:, 4:]

    # Scale 3: stride=32
    stride3 = 32
    grid_size3 = 20
    pred3 = boxes_data[n1 + n2:]  # (400, 49)
    grid_x, grid_y = np.meshgrid(np.arange(grid_size3), np.arange(grid_size3))
    grid3 = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)
    pred3_xy = sigmoid(pred3[:, :2]) * 2 - 0.5
    pred3_wh = (sigmoid(pred3[:, 2:4]) * 2) ** 2
    cx3 = (grid3[:, 0] + pred3_xy[:, 0]) * stride3
    cy3 = (grid3[:, 1] + pred3_xy[:, 1]) * stride3
    w3 = pred3_wh[:, 0] * stride3
    h3 = pred3_wh[:, 1] * stride3
    x1_3 = cx3 - w3 / 2
    y1_3 = cy3 - h3 / 2
    x2_3 = cx3 + w3 / 2
    y2_3 = cy3 + h3 / 2
    boxes3 = np.stack([x1_3, y1_3, x2_3, y2_3], axis=1)
    class_logits3 = pred3[:, 4:]

    # Combine all scales
    boxes_all = np.concatenate([boxes1, boxes2, boxes3], axis=0)  # (8400, 4)
    class_logits_all = np.concatenate([class_logits1, class_logits2, class_logits3], axis=0)  # (8400, 45)

    # Debugging: Check box validity
    print(f"Decoded boxes min/max: {boxes_all.min(axis=0)}, {boxes_all.max(axis=0)}")
    return boxes_all, class_logits_all

def postprocess_output(outputs, conf_thres=0.55, iou_thres=0.5, img_size=(640, 640), orig_size=None):
    raw_output = outputs[0]
    print(f"Raw ONNX output shape: {raw_output.shape}")

    boxes_data = np.transpose(raw_output, (0, 2, 1))[0]
    print(f"Parsed boxes data shape: {boxes_data.shape}")

    # Decode predictions
    boxes_decoded, class_logits_all = decode_predictions(boxes_data, img_size=img_size[0])

    # Apply sigmoid to class logits
    class_scores = sigmoid(class_logits_all)
    scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    print(f"Max class score: {np.max(class_scores)}")
    print(f"Max detection score: {np.max(scores)}")

    # Confidence filtering
    mask = scores > conf_thres
    print(f"Number of boxes after confidence filtering: {np.sum(mask)}")
    boxes_filtered = boxes_decoded[mask]
    scores_filtered = scores[mask]
    class_ids_filtered = class_ids[mask]

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes_filtered.tolist(), scores_filtered.tolist(), conf_thres, iou_thres)

    detections = []
    if len(indices) > 0:
        scale_x = orig_size[1] / img_size[1]
        scale_y = orig_size[0] / img_size[0]
        for i in indices.flatten():
            box = boxes_filtered[i]
            score = scores_filtered[i]
            class_id = int(class_ids_filtered[i])
            class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
            # Scale boxes to original image size
            x_min = box[0] * scale_x
            y_min = box[1] * scale_y
            x_max = box[2] * scale_x
            y_max = box[3] * scale_y
            box_scaled = clamp_box([x_min, y_min, x_max, y_max], orig_size)
            detections.append({
                "box": box_scaled,
                "score": float(score),
                "class_id": class_id,
                "class_name": class_name
            })
            print(f"Raw box: {box}, Scaled box: {box_scaled}")
    return detections

def detect_image(image_path, model_path="sample.onnx"):
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_data, orig_img, orig_size = preprocess_image(image_path)

    outputs = session.run(None, {input_name: input_data})
    print(f"ONNX output shape: {outputs[0].shape}")

    detections = postprocess_output(outputs, conf_thres=0.55, iou_thres=0.5, img_size=(640, 640), orig_size=orig_size)

    if detections:
        for det in detections:
            box = det["box"]
            score = det["score"]
            class_name = det["class_name"]
            print(f"Class: {class_name}, Score: {score:.2f}, Box: {box}")
            cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label = f"{class_name}: {score:.2f}"
            cv2.putText(orig_img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        print("No detections")

    cv2.imshow("Detection", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections

if __name__ == "__main__":
    image_path = "TestFiles/3923.jpg_wh860.jpg"
    model_path = "models/sample.onnx"
    detect_image(image_path, model_path)