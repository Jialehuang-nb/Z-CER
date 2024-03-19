import torch
import os
import cv2
import matplotlib.pyplot as plt
from retinaface import RetinaFace
import numpy as np
video_root="C:/Users/67455/Desktop/abaw-test/videos/"
image_root="C:/Users/67455/Desktop/abaw-test/images_raw/"
save_root="C:/Users/67455/Desktop/abaw-test/images_aligned/"
p = ["05","16","24","28","29","33","40","43","44","51","56"]
def apart(video_path, video_name):
    video = os.path.join(video_path, video_name)
    image_path = os.path.join(image_root, video_name[:-4])
    print(image_path)
    frameFrequency = 1
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    use_video = cv2.VideoCapture(video)
    count = 0
    print('Start extracting images!')
    while True:
        temp = image
        res, image = use_video.read()
        count += 1

        if not res:
            print('not res , not image')
            break

        image_name = os.path.join(image_path,str(count))
        cv2.imwrite(image_name+ '.jpg', image)
        print(image_path + str(count) + '.jpg')
    if video_name in p:
        image_name = os.path.join(image_path,str(count))
        cv2.imwrite(image_name+ '.jpg', temp)
    print('End of image extraction!')
    use_video.release()

def copyface(exist,facedict):
    print(facedict)
    print(exist)
    facesid = [x for x in facedict.keys()]
    for i in range(len(exist)):
        if(exist[i]==0):
            matrix=np.array([abs(i+1-x) for x in facesid])
            id = np.argmin(matrix)
            exist[i]=facesid[id]
    return exist,facedict
'''
#fix in RetinaFace line 239
maxscore=0
    finalface = {}
    for face in obj:
        print(face)
        if(obj[face]["score"]>maxscore):
            maxscore = obj[face]["score"]
            finalface = {'face_1':obj[face]}
    obj = finalface
'''
def cropandalign(file):
    image_path = os.path.join(image_root,file[:-4])
    imgs = os.listdir(image_path)
    exist = np.zeros(len(imgs))
    facedict={}
    for i,img in enumerate(imgs):
        print(i)
        id = int(img[:-4])
        path = os.path.join(image_path,img)
        faces = RetinaFace.extract_faces(img_path = path, align = True)
        for face in faces:
            exist[id-1]=id
            facedict[id]=face
    return exist,facedict

def saveface(file,exist,facedict):
    savepath=os.path.join(save_root,file[:-4])
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for i in range(len(exist)):
        id=i+1
        imgname=savepath+'/'+str(id)+'.jpg'
        plt.imsave(imgname,facedict[exist[i]])
    return

def preprecessing():
    filenames=os.listdir(video_root)
    for file in filenames:
        apart(video_root,file)
        exist,facedict=cropandalign(file)
        exist,facedict=copyface(exist,facedict)
        saveface(file,exist,facedict)
    return

if __name__ == "__main__":
    preprecessing()
    print("end")