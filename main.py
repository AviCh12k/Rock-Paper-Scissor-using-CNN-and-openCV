import cv2
# import tensorflow as tf
import numpy as np
# Categories = ["Paper","Rock", "Scissor"]

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
def countdown(t): 
	
	while t: 
		mins, secs = divmod(t, 60) 
		timer = '{:02d}:{:02d}'.format(mins, secs) 
		print(timer, end="\r") 
		time.sleep(1) 
		t -= 1
	
	print('Fire in the hole!!') 


# input time in seconds 
# t = input("Enter the time in seconds: ") 

# function call 
countdown(int(t)) 





# =============================================================================
cap = cv2.VideoCapture(0)
while True:
    _,frame = cap.read()
    h,w,_ = frame.shape
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # roi_mask = np.zeros([h,w])
    # print(h,w)
    roi = frame[:250,:250]
    cv2.imshow("mask",roi)    
    frame = cv2.rectangle(frame,(0,0),(250,250), (255, 0, 0), 2)
    cv2.imshow("out",frame)
    # prediction = model.predict([prepare(frame)])
    # if 1.0 in prediction[0]:
    #     print(Categories[list(prediction[0]).index(1.0)])


    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release()
cv2.destroyAllWindows()
# =============================================================================
