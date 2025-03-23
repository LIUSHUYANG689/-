# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox, QPushButton, QLabel
from PyQt5.QtGui import *
import cv2
import os
import math
import random
from scipy import misc, ndimage
import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
import imutils
from matplotlib import rcParams
#未使用resize
#答题卡5公务员进行了膨胀操作，不需要Hoffman，
#与答题卡三不同

def order_points(pts):  # 对4点进行排序
    # 共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")
    # 按顺序找到对应坐标0、1、2、3分别是左上、右上、右下、左下
    # 计算左上和右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def toushi_transform(image, pts):  # 输入原始图像和4角点坐标
    # 获取输入坐标点
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # 计算变换矩阵
    print("原始四点坐标:\n", rect, "\n变换后四角点坐标：\n", dst)
    M = cv2.getPerspectiveTransform(rect, dst)
    print("变换矩阵：", M)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # 返回变换后结果
    return warped


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 1200)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # 打开图片
        self.pushButton_openImage = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_openImage.setGeometry(QtCore.QRect(110, 100, 300, 80))
        self.pushButton_openImage.setObjectName("pushButton_openImage")
        # 显示图片区域
        self.label_image = QtWidgets.QLabel(self.centralwidget)
        self.label_image.setGeometry(QtCore.QRect(600, 180, 400, 500))
        self.label_image.setFrameShape(QtWidgets.QFrame.Box)
        self.label_image.setObjectName("label_image")
        self.label_image.setScaledContents(True)  # 图片填充整个框

        # 裁剪按钮
        self.pushButton_crop = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_crop.setGeometry(QtCore.QRect(110, 225, 300, 80))
        self.pushButton_crop.setObjectName("pushButton_crop")

        self.pushButton_HoffmanImage = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_HoffmanImage.setGeometry(QtCore.QRect(110, 350, 300, 80))
        self.pushButton_HoffmanImage.setObjectName("pushButton_saveImage")
        self.pushButton_CannyImage = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_CannyImage.setGeometry(QtCore.QRect(110, 475, 300, 80))
        self.label_image1 = QtWidgets.QLabel(self.centralwidget)
        self.label_image1.setGeometry(QtCore.QRect(1050, 180, 400, 500))
        self.label_image1.setFrameShape(QtWidgets.QFrame.Box)
        self.label_image1.setObjectName("label_image")
        self.label_image1.setScaledContents(True)  # 图片填充整个框

        self.pushButton_Positioning_options = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_Positioning_options.setGeometry(QtCore.QRect(110, 600, 300, 80))
        self.pushButton_Positioning_options.setObjectName("pushButton_Positioning_options")
        self.pushButton_show_data = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_show_data.setGeometry(QtCore.QRect(110, 725, 300, 80))
        self.pushButton_show_data.setObjectName("pushButton_openDirectory")

        self.pushButton_SaveAnswer = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_SaveAnswer.setGeometry(QtCore.QRect(110, 850, 300, 80))
        self.label_text = QtWidgets.QTextEdit(self.centralwidget)
        self.label_text.setGeometry(QtCore.QRect(600, 800, 850, 200))
        self.label_text.setObjectName("label_text")
        self.label_text.setFont(QFont('苏新诗柳楷繁', 15))

        self.text_label0 = QtWidgets.QLabel(self.centralwidget)
        self.text_label0.setGeometry(QtCore.QRect(540, 10, 600, 60))
        self.text_label0.setObjectName("text_label")

        self.text_label1 = QtWidgets.QLabel(self.centralwidget)
        self.text_label1.setGeometry(QtCore.QRect(600, 100, 100, 50))
        self.text_label1.setObjectName("text_label")

        self.text_label2 = QtWidgets.QLabel(self.centralwidget)
        self.text_label2.setGeometry(QtCore.QRect(1050, 100, 300, 50))
        self.text_label2.setObjectName("text_label")

        self.text_label3 = QtWidgets.QLabel(self.centralwidget)
        self.text_label3.setGeometry(QtCore.QRect(600, 700, 300, 50))
        self.text_label3.setObjectName("text_label")

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 50))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton_openImage.clicked.connect(self.openImage)

        self.pushButton_crop.clicked.connect(self.cropImage)

        self.pushButton_HoffmanImage.clicked.connect(self.HoffmanImage)
        self.pushButton_CannyImage.clicked.connect(self.Canny_img)
        self.pushButton_Positioning_options.clicked.connect(self.Positioning_options)
        self.pushButton_show_data.clicked.connect(self.show_data)
        self.pushButton_SaveAnswer.clicked.connect(self.SaveAnswer)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

        self.pushButton_openImage.setText(_translate("MainWindow", "打开图像"))

        self.pushButton_openImage.setFont(QFont('苏新诗柳楷繁', 15))
        self.label_image.setText(_translate("MainWindow", ""))

        self.pushButton_crop.setText(_translate("MainWindow", "裁剪图像"))
        self.pushButton_crop.setFont(QFont('苏新诗柳楷繁', 15))

        self.pushButton_HoffmanImage.setText(_translate("MainWindow", "霍夫曼矫正"))
        self.pushButton_HoffmanImage.setFont(QFont('苏新诗柳楷繁', 15))
        self.pushButton_CannyImage.setText(_translate("MainWindow", "Canny边缘检测"))
        self.pushButton_CannyImage.setFont(QFont('苏新诗柳楷繁', 15))
        self.label_image1.setText(_translate("MainWindow", ""))
        self.pushButton_Positioning_options.setText(_translate("MainWindow", "选项定位"))
        self.pushButton_Positioning_options.setFont(QFont('苏新诗柳楷繁', 15))
        self.text_label0.setText(_translate("MainWindow", "答题卡识别系统"))
        self.text_label0.setFont(QFont('苏新诗柳楷繁', 25))
        self.text_label1.setText(_translate("MainWindow", "原图:"))
        self.text_label1.setFont(QFont('苏新诗柳楷繁', 15))
        self.text_label2.setText(_translate("MainWindow", "处理后:"))
        self.text_label2.setFont(QFont('苏新诗柳楷繁', 15))
        self.text_label3.setText(_translate("MainWindow", "识别答案:"))
        self.text_label3.setFont(QFont('苏新诗柳楷繁', 15))
        self.pushButton_show_data.setText(_translate("MainWindow", "答案输出"))
        self.pushButton_show_data.setFont(QFont('苏新诗柳楷繁', 15))
        self.pushButton_SaveAnswer.setText(_translate("MainWindow", "保存答案"))
        self.pushButton_SaveAnswer.setFont(QFont('苏新诗柳楷繁', 15))
        # self.label_directoryPath.setText(_translate("MainWindow", "文件夹路径"))

    def __init__(self):
        self.is_cropped = False  # 初始化裁剪标志
        self.is_hoffman_applied = False
        self.crop_image_path = r"D:\\pic\\cut_image.png"  # 裁剪图像保存路径
        self.original_image_path = None  # 原始图像

    def openImage(self):  # 选择本地图片上传
        global imgName  # 这里为了方便别的地方引用图片路径，我们把它设置为全局变量
        try:
            imgName, imgType = QFileDialog.getOpenFileName(self.centralwidget, "打开图片", "",
                                                           "Image Files (*.png *.jpg);;All Files(*)")  # 弹出一个文件选择框
            if imgName:  # 确保用户选择了文件
                self.original_image_path = imgName
                jpg = QtGui.QPixmap(imgName).scaled(self.label_image.width(),
                                                    self.label_image.height())  # 通过文件路径获取图片文件，并设置图片长宽为label控件的长宽
                self.label_image.setPixmap(jpg)  # 在label控件上显示选择的图片
                self.is_cropped = False
        except Exception as e:
            QMessageBox.critical(self.centralwidget, "错误", f"无法打开图像: {e}")
            print(e)
        # 读取图像
        self.img = cv2.imread(imgName)
        self.showImageForCropping()

    def showImageForCropping(self):  # 显示原始图像并激活裁剪功能
        cv2.namedWindow('image')
        cv2.imshow('image', self.img)
        cv2.setMouseCallback('image', self.on_mouse)

    def cropImage(self):  # 启动裁剪功能
        self.showImageForCropping()

    def on_mouse(self, event, x, y, flags, param):
        global point1, point2, img
        img2 = self.img.copy()

        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            point1 = (x, y)
            cv2.circle(img2, point1, 10, (0, 255, 0), 5)
            cv2.imshow('image', img2)

        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
            cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 5)
            cv2.imshow('image', img2)

        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
            point2 = (x, y)
            cv2.rectangle(img2, point1, point2, (0, 0, 255), 5)
            cv2.imshow('image', img2)

            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])

            # 裁剪图像
            cut_img = self.img[min_y:min_y + height, min_x:min_x + width]
            cv2.imshow('cut_img', cut_img)

            # 显示裁剪的图像
            cut_image_path = r"D:\\pic\\cut_image.png"
            cv2.imwrite(cut_image_path, cut_img)
            print(f"裁剪图像已保存至: {self.crop_image_path}")
            self.is_cropped = True
            # 更新界面
            cut_img = QtGui.QPixmap(cut_image_path).scaled(self.label_image1.width(), self.label_image1.height())
            self.label_image1.setPixmap(cut_img)

    def HoffmanImage(self):  # 保存图片到本地
        def Hoffman(img, gray):
            global Hoffman_img
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            # 霍夫变换
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                if x1 == x2 or y1 == y2:
                    print("不用霍夫曼直线变换")
                    return img
                t = float(y2 - y1) / (x2 - x1)
                rotate_angle = math.degrees(math.atan(t))
                if rotate_angle > 45:
                    rotate_angle = -90 + rotate_angle
                elif rotate_angle < -45:
                    rotate_angle = 90 + rotate_angle
                rotate_img = ndimage.rotate(img, rotate_angle)
                return rotate_img

        image_path = self.crop_image_path if self.is_cropped else self.original_image_path
        if not image_path:
            QMessageBox.warning(self.centralwidget, "警告", "请先上传图像！")
            return

        img = cv2.imread(image_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rotate_img = Hoffman(img, gray)
        cv2.imwrite(r"D:\\pic\\random_image.png", rotate_img)

        self.is_hoffman_applied = True
        Hoffman_img = QtGui.QPixmap(r"D:\\pic\\random_image.png").scaled(self.label_image1.width(),
                                                                         self.label_image1.height())
        self.label_image1.setPixmap(Hoffman_img)

    def Canny_img(self):  # 边缘检测
        global Hoffman_img, paper, warped

        if self.is_hoffman_applied:  # 使用霍夫曼变换后的图像
            img_path = r"D:\\pic\\random_image.png"
        elif self.is_cropped:  # 使用裁剪后的图像
            img_path = self.crop_image_path
        elif self.original_image_path:  # 使用原始图像
            img_path = self.original_image_path
        else:
            QMessageBox.warning(self.centralwidget, "警告", "请先上传图像或进行裁剪！")
            return
        # 尝试读取 "random_image.png"
        Hoffman_img = cv2.imread(img_path)




        gray = cv2.cvtColor(Hoffman_img, cv2.COLOR_BGR2GRAY)

        #thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # 膨胀操作连接断开的部分（对于开放轮廓）
        # kernel = np.ones((3, 3), np.uint8)
        # dilated = cv2.dilate(edged, kernel, iterations=1)
        # cv2.imshow("Dilated Edges", dilated)

        #cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # 灰度化
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        # canny边缘检测
        edged = cv2.Canny(blurred, 50, 150)
        cv2.namedWindow('canny',0)
        cv2.imshow("canny",edged)
        # 使用自适应阈值处理图像，以连接边缘

        #adaptive_thresh = cv2.adaptiveThreshold( edged, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        #cv2.imshow("canny2",adaptive_thresh)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edged, kernel, iterations=2)
        #eroded = cv2.erode(edged, kernel, iterations=1)
        #opened = cv2.dilate(eroded, kernel, iterations=1)
        #eroded = cv2.erode(dilated, kernel, iterations=1)
        #cv2.imshow("Dilated Edges", dilated)
        # 查找轮廓
        #cv2.imshow('canny2', eroded)
        #closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        cnts, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        canny_img = cv2.drawContours(Hoffman_img, cnts, -1, (0, 255, 0), 3)
        output_path = r"D:\\pic\\canny_contours.png"
        cv2.imwrite(output_path, canny_img)
        # 加载保存的图像到界面显示
        cv2.imshow("canny_img", canny_img)

        print(f"轮廓图像已保存至: {output_path}")

        docCnt = []
        count = 0
        # 确保至少有一个轮廓被找到

        if len(cnts) > 0:
        # 将轮廓按照大小排序
             cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
          # 对排序后的轮廓进行循环处理
             for c in cnts:
          # 获取近似的轮廓
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                  docCnt.append(approx)
                  count += 1
                  if count == 3:
                    break


        # 画出四点区域
        if not docCnt:
            print("No contours with four points found")
        else:

            # 确保docCnt不为空
            if docCnt:
                # 遍历docCnt中的每个轮廓
                for contour in docCnt:
                    # 在canny_img上绘制轮廓，颜色为红色，线宽为2
                    cv2.drawContours(Hoffman_img, [contour], -1, (0, 0, 255), 2)
                # 显示绘制了轮廓的图像
                cv2.imshow("canny_img_with_docCnt", Hoffman_img)
                # 保存绘制了轮廓的图像

                cv2.imwrite(r"D:\pic\Hoffman.png", Hoffman_img)
                print(f"带有docCnt轮廓的图像已保存至")
            else:
                print("没有找到符合条件的轮廓")

        # 绘制角点

        paper = toushi_transform(Hoffman_img, np.array(docCnt[0]).reshape(4, 2))
        warped = toushi_transform(gray, np.array(docCnt[0]).reshape(4, 2))

        cv2.imshow('warped', warped)
        cv2.imwrite(r"D:\\pic\\paper.png", paper)
        paper = QtGui.QPixmap(r"D:\\pic\\paper.png").scaled(self.label_image1.width(),
                                                        self.label_image1.height())
        self.label_image1.setPixmap(paper)
    def Positioning_options(self):  # 保存文本文件
        global Hoffman_img, warped, ID_Answer

        def judgeX(x, y, mode):
            if mode == "point":
                if y < 400:
                    if x < 600:
                        return int((x - 34) / 100) + 1
                    elif x < 1200:
                        return int((x - 600) / 100) + 6
                    elif x < 1700:
                        return int((x - 1200) / 100) + 11
                    else:
                        return int((x - 1700) / 110) + 16
                elif y < 800:
                    if x < 635:
                        return int(x / 100) + 21
                    elif x < 1270:
                        return int((x - 635) / 100) + 26
                    elif x < 1890:
                        return int((x - 1270) / 100) + 31
                    else:
                        return int((x - 1890) / 100) + 36
                elif y < 1200:
                    if x < 635:
                        return int(x / 100) + 41
                    elif x < 1270:
                        return int((x - 635) / 100) + 46
                    elif x < 1890:
                        return int((x - 1270) / 100) + 51
                    else:
                        return int((x - 1890) / 100) + 56
                else:
                    return False


        def judgeY(y, mode):
            if mode == "point":
                if 0 < (y % 370) <= 130:
                    return 'A'
                elif 130 < y % 370 <= 200:
                    return 'B'
                elif 200 < y % 370 <= 270:
                    return 'C'
                elif 270 < y % 370 <= 370:
                    return 'D'
                else:
                    return False

        def judge(x, y, mode):
            if judgeY(y, mode) != False and judgeX(x,y, mode) != False:
                if mode == "point":
                    return judgeX(x, y, mode), judgeY(y, mode)
            else:
                return 0
        def judge_point(answers, mode):
            IDAnswer = []
            for answer in answers:
                if (judge(answer[0], answer[1], mode)) != 0:
                    IDAnswer.append(judge(answer[0], answer[1], mode))
                else:
                    continue
            IDAnswer.sort()
            return IDAnswer

        paper = cv2.imread(r"D:\\pic\\paper.png")
        # 二值化处理
        thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        ChQImg = cv2.blur(thresh, (28, 25))
        ChQImg = cv2.threshold(ChQImg, 100, 225, cv2.THRESH_BINARY)[1]
        cv2.imshow('ChQImg',ChQImg)
        cv2.imwrite(r"D:\\pic\\Chqimg.png", ChQImg)
        cnts1, hierarchy = cv2.findContours(ChQImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        questionCnts = []
        answers = []
        # 对每一个轮廓进行循环处理
        for c in cnts1:
            # 计算轮廓的边界框，然后利用边界框数据计算宽高比,
            # x和y表示矩形左上角的坐标，w和h表示矩形的宽和高
            # print(c)
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            # 判断轮廓是否是答题框

            if  (w >= 40) and h >= 30 and 1.2 <= ar <= 2:
                # print(x, y, w, h)
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                questionCnts.append(c)
                answers.append((cX, cY))
                cv2.circle(warped, (cX, cY), 7, (255, 255, 255), -1)
        ID_Answer = judge_point(answers, mode="point")
        cv2.drawContours(warped, questionCnts, -1, (255, 0, 0), 3)
        warped = cv2.resize(warped, (2400, 2800), cv2.INTER_LANCZOS4)
        cv2.imwrite('D:\\pic\\paper1.jpg', paper)
        cv2.imwrite('D:\\pic\\warped1.jpg', warped)
        paper1 = QtGui.QPixmap('D:\\pic\\warped1.jpg').scaled(self.label_image1.width(),
                                                           self.label_image1.height())
        self.label_image1.setPixmap(paper1)

    def show_data(self):  # 保存选项信息

        global ID_Answer, m
        m = " "
        for i in ID_Answer:
            if int(i[0]) % 5 == 0:
                s = str(i[0]) + " : " + str(i[1]) + ": " + "\n"
            else:
                s = str(i[0]) + " : " + str(i[1]) + ": "
            m = m + s

        self.label_text.setPlainText(str(m))

    def SaveAnswer(self):
        global m

        fd, fp = QFileDialog.getSaveFileName(self.centralwidget, "保存文件", "", "*.txt;;All Files(*)")
        f = open(fd, 'w')
        f.write(m)
        f.close()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    formObj = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(formObj)
    formObj.show()
    sys.exit(app.exec_())