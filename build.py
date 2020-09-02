import keyboard
import cv2 
import os 

cat = input("Enter Category: ")

directory = r'D:\stonepaperscissor\Dataset3'
cap = cv2.VideoCapture(0)
i=0
while True:
    if i==2000:
        break
    _,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = gray[:250,:250]
    cv2.imshow("mask",roi)    
    frame = cv2.rectangle(img,(0,0),(250,250), (255, 0, 0), 2)
    img = cv2.putText(img, str(i), (50,450), cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 0, 255), thickness=3)
    cv2.imshow("out",img)

    os.chdir(directory+ '\\' + cat) 
    filename = cat + str(i) + '.jpg'

    if keyboard.is_pressed('c'):  # if key 'q' is pressed 
        # print('You Pressed A Key! ' + str(i))
        cv2.imwrite(filename, roi) 
        i=i+1
        
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release()
cv2.destroyAllWindows()


