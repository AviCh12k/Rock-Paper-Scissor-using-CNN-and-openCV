import cv2
# import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
Categories = ["Paper","Rock", "Scissor"]

# def prepare(frame):
#         img_size = 128
#         img_array = frame
#         new_array = cv2.resize(img_array,(img_size,img_size))
#         return new_array.reshape(-1, img_size, img_size, 3)

# model = tf.keras.models.load_model("model.h5")
# for x in range(1,8):
#         img = cv2.imread("test"+str(x)+".jpg")
#         prediction = model.predict([prepare(gray)])
#         print(Categories[list(prediction[0]).index(1.0)])





# import the time module 
import time 

# define the countdown func. 
# def countdown(t,frame): 
	
# 	while t: 
# 		mins, secs = divmod(t, 60) 
# 		timer = '{:02d}:{:02d}'.format(mins, secs) 
#         frame = cv2.putText(frame, str(timer), (50,450), cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 0, 255), thickness=3)
		# print(timer, end="\r") 
		# time.sleep(1) 
		# t -= 1
	# print('Fire in the hole!!') 

# input time in seconds 
# t = input("Enter the time in seconds: ") 

# function call 
# countdown(int(t)) 





# =============================================================================
time = "3"
cap = cv2.VideoCapture(0)
i=0
while True:
    _,frame = cap.read()
    h,w,_ = frame.shape
    # print(h,w)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # roi_mask = np.zeros([h,w])
    # print(h,w)
    roi = frame[:250,:250]
    cv2.imshow("mask",roi)    
    frame = cv2.rectangle(frame,(0,0),(250,250), (255, 0, 0), 2)
    if i<20:
        count = 0
        frame = cv2.putText(frame, "3", (300,100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 0, 255), thickness=3)  
    elif i<40:
        frame = cv2.putText(frame, "2", (300,100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 0, 255), thickness=3) 
    elif i<60:
        frame = cv2.putText(frame, "1", (300,100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 0, 255), thickness=3)
        x = random.randint(0, 2)
    elif i<75:
        frame = cv2.putText(frame, "GO", (300,100), cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 0, 255), thickness=3)
        comp_img = cv2.imread("Computer/" + Categories[x] + ".jpg")
        # comp_img = cv2.cvtColor(comp_img,cv2.COLOR_BGR2RGB)
        # cv2.imshow("aa",comp_img)
        h,w,_ = comp_img.shape
        # print(h,w)
        frame[50:178, 500:628] = comp_img
        if count == 0:
            capture = roi
            count+=1
            # plt.imshow(capture)
            # plt.show()
    else:
        i=0
    cv2.imshow("out",frame)
    # prediction = model.predict([prepare(frame)])
    # if 1.0 in prediction[0]:
    #     print(Categories[list(prediction[0]).index(1.0)])
    i+=1

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release()
cv2.destroyAllWindows()
# =============================================================================
