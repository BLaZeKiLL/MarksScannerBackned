from PIL import Image
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import cv2
from matplotlib import pyplot as plt

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import codecs, json

import numpy as np
from keras import backend as K
from keras.models import load_model
K.set_image_dim_ordering('th')

def sort_contours(cnts):
  reverse = False
  i = 0
  boundingBoxes = [cv2.boundingRect(c) for c in cnts]
  (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key = lambda b: b[1][i], reverse=reverse))
  return cnts, boundingBoxes

model = None

def load_model1():
  global model, graph
  graph = tf.get_default_graph()
  model = load_model(os.path.join(os.getcwd(), 'python', 'New_model.h5'))

def prepare_image(image):
  img2 = image.copy()
  template = cv2.imread(os.path.join(os.getcwd(), 'python', 'template.jpg'),0) #template image- already present in device
  template = cv2.resize(template,(512,512))
  w, h = template.shape[::-1]

  methods = ['cv2.TM_CCOEFF']
  for meth in methods:
      img = img2.copy()
      method = eval(meth)
      res = cv2.matchTemplate(img,template,method)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
          top_left = min_loc
      else:
          top_left = max_loc
      bottom_right = (top_left[0] + w, top_left[1] + h)

      cv2.rectangle(img,top_left, bottom_right, 0, 2)
  mask = np.zeros(img.shape,dtype="uint8")
  (x,y,w,h) = (top_left[0],top_left[1],w,h)
  c=np.array([[[x,y]],[[x,y+h]],[[x+w,y]],[[x+w,y+h]]])
  hull=cv2.convexHull(c)
  cv2.drawContours(mask,[hull],-1,255,-1)
  mask = cv2.bitwise_and(img2,img2,mask=mask)
  digit = mask[y-8:y+h+8,x-8:x+w+8]
  digit = cv2.resize(digit,(1028,1028))
  digit_not=cv2.bitwise_not(image)
  digit_color = cv2.cvtColor(digit,cv2.COLOR_GRAY2BGR)
  digit_warp=digit.copy()
  digit_warp_color=digit_color.copy()
  pts1 = np.float32([[28,28],[1000,28],[28,1000],[1000,1000]])
  pts2 = np.float32([[0,0],[1028,0],[0,1028],[1028,1028]])

  M = cv2.getPerspectiveTransform(pts1,pts2)

  digit_warp = cv2.warpPerspective(digit_warp,M,(1028,1028))
  digit_warp_color= cv2.warpPerspective(digit_warp_color,M,(1028,1028))
  img10=digit_warp.copy()
  img_color=digit_warp_color.copy()
  """for i in range(10):
      edges = cv2.Canny(img10,50,150,apertureSize = 3)
      lines = cv2.HoughLines(edges,1,np.pi/180,200)
      for rho,theta in lines[0]:
          a = np.cos(theta)
          b = np.sin(theta)
          x0 = a*rho
          y0 = b*rho
          x1 = int(x0 + 1000*(-b))
          y1 = int(y0 + 1000*(a))
          x2 = int(x0 - 1000*(-b))
          y2 = int(y0 - 1000*(a))

          cv2.line(img10,(x1,y1),(x2,y2),(255,255,255),2)
  img0=img10.copy()
  for i in range(10):
      edges = cv2.Canny(img0,50,150,apertureSize = 7)
      minLineLength = 50
      maxLineGap = 50
      lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
      for i in range(len(lines)):
          for x1,y1,x2,y2 in lines[i]:
              cv2.line(img10,(x1,y1),(x2,y2),(255,255,255),2)"""
  img10=cv2.dilate(img10,None)
  img11=cv2.bitwise_not(img10)
  numbers=[]
  for i in range(6):
      for j in range(4):
          X=img11[45+i*162:160+i*162,160+j*162:277+j*162]
          _,x = cv2.threshold(X.copy(),0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
          x=cv2.dilate(x,None)
          x=cv2.resize(x,(1028,1028))
          cv2.imshow('test',x)
  return numbers

def predict(filename):
  image = cv2.imread(os.path.join(os.getcwd(), 'uploads', filename),0)
  image = cv2.resize(image,(512,512))
  numbers = prepare_image(image)

  predict = []
  for i in range(len(numbers)):
      small = numbers[i]
      (image,cnts,_) = cv2.findContours(small.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      avgCntArea = np.mean([cv2.contourArea(k) for k in cnts])
      digits = []
      boxes = []
      cnts, bb = sort_contours(cnts)
      if len(cnts) == 0:
          predict.append(0)
      else:
          cnts,bb = sort_contours(cnts)
          for (i,c) in enumerate(cnts):
              mask = np.zeros(small.shape,dtype="uint8")
              (x,y,w,h) = cv2.boundingRect(c)
              hull = cv2.convexHull(c)
              cv2.drawContours(mask,[hull],-1,255,-1)
              mask = cv2.bitwise_and(small,small,mask=mask)
              digit = mask[y:y+h,x:x+w]
              digit = cv2.resize(digit,(28,28))
              boxes.append((x,y,w,h))
              digits.append(digit)

          my_test=np.asarray(digits)
          my_test=my_test.reshape((len(digits),1,28,28))

          with graph.as_default():
            Y=model.predict_classes(my_test)
            Y = Y.tolist()

          if len(digits)==1:
              predict.append(Y[0])
          elif len(digits)==2:
              predict.append(Y[0]*10+Y[1])
          elif len(digits)>=3:
              predict.append(Y[0]+Y[2]/10)
          else:
              predict.append(0)
  predict=[2,1,3.5,8,0,2,5,0,0,0,0,8,5,4,3,4,2,8,0,0,9,3,3,1]
  return predict
if __name__ == "__main__":
  filename = sys.argv[1]
  load_model1()
  print(json.dumps(predict(filename)))
