#coding=utf-8
# shutil的解释： os模块不仅提供了新建文件、删除文件、查看文件属性的操作功能，
# 还提供了对文件路径的操作功能。但是，对于移动、复制、打包、压缩、解压文件
# 及文件夹等操作，os模块没有提供相关的函数，此时需要用到shutil模块。shutil
# 模块是对os模块中文件操作的补充，是Python自带的关于文件、文件夹、压缩文件
# 的高层次的操作工具，类似于高级API。

import numpy as np
import cv2
import shutil 
import os


# 生成自己的人脸数据
def generator(data):
    name = input('Input Name: ')
    # 如果路径存在则删除
    path = os.path.join(data, name)
    if os.path.isdir(path):
        shutil.rmtree(path) #递归删除文件夹
    # 创建文件
    os.mkdir(path)
    # 创建一个级联分类器，加载一个xml分类器文件，它既可以是Harr特征也可以是LBP特征的分类器
    face_cascade = cv2.CascadeClassifier('F:\\face_recognize\\haarcascade_frontalface_default.xml')
    # 打开摄像头
    camera = cv2.VideoCapture(1)
    cv2.namedWindow('Face')
    # 计数
    count = 1
    while True:
        # 读取一帧图像
        if count > 10:
            break
        ret, frame = camera.read()
        # 判断图片是否读取成功
        if ret:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #人脸检测
            faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
            for (x,y,w,h) in faces:
                # 在原图像上绘制矩形
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                # 调整图像大小 和ORL人脸库图像一样大小
                f = cv2.resize(frame[y:y+h,x:x+w],(92,112))
                # 保存人脸
                cv2.imwrite('%s/%s.bmp'%(path,str(count)),f)
                count += 1
                if(count > 10):
                    break
            cv2.imshow('Face', frame)
            #如果按下q键则退出
            if cv2.waitKey(100) & 0xff == ord('q') :
                break
    camera.release()
    cv2.destroyAllWindows()

# 读取
def LoadData(data):
    # data表示训练数据集所在的目录，要求图片尺寸一致
    # images：[m, height, width] 其中m代表样本个数，height代表图片高度，width代表宽度
    # names: 名字的集合
    # labels: 标签
    
    images = []
    labels = []
    names = []
    
    label = 0
    #过滤所有的文件夹
    for subDirname in os.listdir(data):
        subjectPath = os.path.join(data,subDirname)
        if os.path.isdir(subjectPath):                
            #每一个文件夹下存放着一个人的照片    
            names.append(subDirname)
            for fileName in os.listdir(subjectPath):
                imgPath = os.path.join(subjectPath,fileName)
                img = cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
                images.append(img)
                labels.append(label)
            label += 1
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images,labels,names
    
def FaceRecognize():    
    #加载训练数据
    X,y,names=LoadData('F:\\ORL')
    
    model = cv2.face.EigenFaceRecognizer_create()
    model.train(X,y)
    
    face_cascade = cv2.CascadeClassifier('F:\\face_recognize\\haarcascade_frontalface_default.xml')
    
    #打开摄像头    
    camera = cv2.VideoCapture(1)
    cv2.namedWindow('Face')
    
    while(True):
        # 读取一帧图像
        ret,frame = camera.read()
        #判断图片读取成功
        if ret:
            gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #人脸检测
            
            faces = face_cascade.detectMultiScale(gray_img,1.3,5)            
            for (x,y,w,h) in faces:
                #在原图像上绘制矩形
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray_img[y:y+h,x:x+w]
                try:
                    #宽92 高112
                    roi_gray = cv2.resize(roi_gray,(92,112),interpolation=cv2.INTER_LINEAR)
                    params = model.predict(roi_gray)
                    print('Label:%s,confidence:%.2f'%(params[0],params[1]))
                    cv2.putText(frame,names[params[0]],(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
                except:
                    continue

            cv2.imshow('Face',frame)            
            #如果按下q键则退出
            if cv2.waitKey(100) & 0xff == ord('q') :
                break
    camera.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    data = 'F:\\ORL'
    generator(data)
    FaceRecognize()