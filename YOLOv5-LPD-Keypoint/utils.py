import cv2
import numpy as np
import colorsys


def four_point_transform(image, pts):
    rect = pts.astype('float32')
    br_x, br_y, bl_x, bl_y, tl_x, tl_y, tr_x, tr_y = rect
    widthA = np.sqrt(((br_x - bl_x) ** 2) + ((br_y - bl_y) ** 2))
    widthB = np.sqrt(((tr_x - tl_x) ** 2) + ((tr_y - tl_y) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr_x - br_x) ** 2) + ((tr_y - br_y) ** 2))
    heightB = np.sqrt(((tl_x - bl_x) ** 2) + ((tl_y - bl_y) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    rect =  np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype='float32')
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def resize_padd(img, size, color=(114, 114, 114)):
    h, w = img.shape[0:2]  # hw
    # resize
    r = size / max(h, w)
    if r != 1:
        interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (round(w * r), round(h * r)), interpolation=interp)
    # padd
    n_h, n_w = img.shape[0:2]
    padd_h = size - n_h
    padd_w = size - n_w
    padd_h /= 2
    padd_w /= 2
    top, bottom = int(round(padd_h - 0.1)), int(round(padd_h + 0.1))
    left, right = int(round(padd_w - 0.1)), int(round(padd_w + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def cxcywh2xyxys(cxcywh, scale, left, top, img_hw):
    out = np.zeros_like(cxcywh)
    x1 = cxcywh[:, 0] - (cxcywh[:, 2] / 2) - left
    x2 = cxcywh[:, 0] + (cxcywh[:, 2] / 2) - left
    y1 = cxcywh[:, 1] - (cxcywh[:, 3] / 2) - top
    y2 = cxcywh[:, 1] + (cxcywh[:, 3] / 2) - top

    out[:, 0] = np.clip(x1/scale, 0, img_hw[1])
    out[:, 1] = np.clip(y1/scale, 0, img_hw[0])
    out[:, 2] = np.clip(x2/scale, 0, img_hw[1])
    out[:, 3] = np.clip(y2/scale, 0, img_hw[0])
    return out

def scale_landmarks(landmarks, scale, left, top, img_hw):
    out = np.zeros_like(landmarks)
    out[:, [0,2,4,6]] = landmarks[:, [0,2,4,6]] - left
    out[:, [1,3,5,7]] = landmarks[:, [1,3,5,7]] - top
    out = np.clip(out/scale, 0, img_hw[1])
    return out


def nms(boxes, confs, iou_thresh=0.3):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = confs.flatten().argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def pre_process(img_raw, img_size=640):
    img_data, scale, padd_data = resize_padd(img_raw, img_size)
    img_data = img_data / 255.0  # 归一化
    img_data = img_data[..., ::-1]  # BGR -> RGB格式
    img_data = np.transpose(img_data, (2, 0, 1))  # HWC to CHW
    img_data = np.expand_dims(img_data, 0)  # to BCHW
    np_data = np.array(img_data, dtype=np.float32)
    return np_data, scale, padd_data


def post_process(pred, conf_thred, iou_thred, scale, left, top, img_hw):
    """
    过滤低置信度结果, 缩放回原图尺寸, nms
    """
    true_conf = pred[:, 4:5] * pred[:, 13:]
    if isinstance(conf_thred, list):  # 不同类可使用不同的置信度
        mask = true_conf >= np.array(conf_thred)
    else:
        mask = true_conf >= conf_thred
    mask = np.sum(mask, axis=-1) > 0
    boxes = cxcywh2xyxys(pred[mask][:, 0:4], scale, left, top, img_hw)
    landmarks = scale_landmarks(pred[mask][:, 5:13], scale, left, top, img_hw)
    confs = np.max(true_conf[mask], axis=-1)
    classes = np.argmax(pred[mask][:, 13:], axis=-1)
    # 所有分类一起做nms
    c = classes.reshape((-1, 1)) * 1000  # 或者是其他大于max([w, h])的数
    nms_boxes = boxes + c
    keep = nms(nms_boxes, confs, iou_thred)
    confs = confs[keep]
    boxes = boxes[keep]
    landmarks = landmarks[keep]
    classes = classes[keep]
    return boxes, landmarks, confs, classes


def get_color_list(label_list):
    # 根据标签生成颜色
    color_list = []
    hues = np.linspace(0, 1, len(label_list) + 1)
    for hue in hues.tolist():
        rgb = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]
        color_list.append(rgb)
    return color_list


def draw_boxes(img, boxes, confs, labels, color, thickness=1):
    for bbox, cls, conf in zip(boxes, labels, confs):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        label = f"{cls}: {conf:.2f}"
        cv2.putText(img, label, (int(x1), int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, thickness / 3, color, thickness)
    return img

def draw_points(img, points, color, thickness=2):
    points_list = []
    for landmarks in points:
        for i in range(len(landmarks) // 2):
            points_list.append((landmarks[2*i].round().astype(np.int32), landmarks[2*i+1].round().astype(np.int32)))
    for i, p in enumerate(points_list):
        cv2.circle(img, p, thickness, color, -1)
    return img



