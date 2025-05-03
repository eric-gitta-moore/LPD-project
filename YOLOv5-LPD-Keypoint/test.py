import cv2
import numpy as np
from utils import plot_one_box

'''
Could not load the Qt platform plugin "xcb"
@see https://github.com/NVlabs/instant-ngp/discussions/300#discussioncomment-8781708
'''
def test_plot_one_box():
    # 创建一个示例图像 (500x500的黑色图像)
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # 定义一个边界框 [x1, y1, x2, y2]
    # 在图像中心画一个100x100的框
    bbox = [200, 200, 300, 300]
    
    # 定义标签文本
    label = "车牌 0.95"
    
    # 定义框的颜色 (B,G,R)
    color = (0, 255, 0)  # 绿色
    
    # 调用plot_one_box函数绘制边界框
    plot_one_box(bbox, img, color=color, label=label, line_thickness=2)
    
    # 显示结果
    cv2.imshow('Test Plot One Box', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    # cv2.imwrite('test_result.jpg', img)

if __name__ == "__main__":
    test_plot_one_box() 