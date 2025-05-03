import cv2
import numpy as np
import colorsys
from PIL import Image, ImageDraw, ImageFont
import os


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

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # font_path = os.path.join(os.path.dirname(__file__), "../simsun.ttc")
    font_path = "simsun.ttc"
    # 字体的格式
    fontStyle = ImageFont.truetype(font_path, textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    """一般会用在detect.py中在nms之后变量每一个预测框，再将每个预测框画在原图上
    使用opencv在原图im上画一个bounding box
    :params x: 预测得到的bounding box  [x1 y1 x2 y2]
    :params im: 原图 要将bounding box画在这个图上  array
    :params color: bounding box线的颜色
    :params labels: 标签上的框框信息  类别 + score
    :params line_thickness: bounding box的线宽
    @see https://github.com/eric-gitta-moore/YOLOv5-LPRNet-Licence-Recognition/blob/3d950d62f7ee519fee8a61c2240e7d2f728c2f58/utils/utils.py#L957
    """
    # check im内存是否连续
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    # tl = 框框的线宽  要么等于line_thickness要么根据原图im长宽信息自适应生成一个
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    # c1 = (x1, y1) = 矩形框的左上角   c2 = (x2, y2) = 矩形框的右下角
    c1, c2 = (int(float(x[0])), int(float(x[1]))), (int(float(x[2])), int(float(x[3])))
    # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle: 在im上画出框框   c1: start_point(x1, y1)  c2: end_point(x2, y2)
    # 注意: 这里的c1+c2可以是左上角+右下角  也可以是左下角+右上角都可以
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # 如果label不为空还要在框框上面显示标签label + score
    if label:
        tf = max(tl - 1, 1)  # label字体的线宽 font thickness
        # cv2.getTextSize: 根据输入的label信息计算文本字符串的宽度和高度
        # 0: 文字字体类型  fontScale: 字体缩放系数  thickness: 字体笔画线宽
        # 返回retval 字体的宽高 (width, height), baseLine 相对于最底端文本的 y 坐标
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # 同上面一样是个画框的步骤  但是线宽thickness=-1表示整个矩形都填充color颜色
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText: 在图片上写文本 这里是在上面这个矩形框里写label + score文本
        # (c1[0], c1[1] - 2)文本左下角坐标  0: 文字样式  fontScale: 字体缩放系数
        # [225, 255, 255]: 文字颜色  thickness: tf字体笔画线宽     lineType: 线样式
        # cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        return cv2AddChineseText(im, label, (c1[0], c1[1]-t_size[1]), tuple([225, 255, 255]), t_size[1])
    return im
