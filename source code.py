import cv2
import numpy as np
import mysql.connector
import time
import datetime

from numpy._typing import _128Bit
"""
def create_connection():
    
    connection = mysql.connector.connect(
        host="HOSTIP",  
        user="userID",  
        port='PORT',
        password="PASSWD",  
        database="DB"  
    )
    return connection

def insert_data(connection, data):
 
    cursor = connection.cursor()

  
    query = "INSERT INTO new_table (name,time, detection) VALUES (%s,%s, %s)"

    cursor.execute(query, data)


    connection.commit()
"""

classNames = { 1: 'person', 77: 'cell phone'}
def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value

def check_overlap(box1, box2):
    x1_max = max(box1[0], box2[0])
    x2_min = min(box1[2], box2[2])
    y1_max = max(box1[1], box2[1])
    y2_min = min(box1[3], box2[3])

    overlap = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
    return overlap > 0


camera = cv2.VideoCapture(0)
camera.set(3, 1280)
camera.set(4, 720)

def main():
    try:
        time1= time.time()
        model = cv2.dnn.readNetFromTensorflow('C:\\Users\\iot22\\source\\repos\\1105_moblienet_v2_test\\1105_moblienet_v2_test\\frozen_inference_graph.pb',
                                              'C:\\Users\\iot22\\source\\repos\\1105_moblienet_v2_test\\1105_moblienet_v2_test\\ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        person_boxes = []
        knife_boxes = []
        while True:
            start_time = time.time()
            _, image = camera.read()
            image_height, image_width, _ = image.shape
            
            model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
            output = model.forward()
            
            for detection in output[0, 0, :, :]:
                confidence = detection[2]
                class_id = int(detection[1])
                
                box_x = detection[3] * image_width
                box_y = detection[4] * image_height
                box_width = detection[5] * image_width
                box_height = detection[6] * image_height

                if class_id == 1 and confidence > 0.7:
                    person_boxes.append([int(box_x), int(box_y), int(box_width), int(box_height)])
                    cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (0, 255, 0), 1)
                    label = f"person: {confidence:.2f}"
                    cv2.putText(image, label, (int(box_x), int(box_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                if class_id == 77 and confidence > 0.7:  
                    knife_boxes.append([int(box_x), int(box_y), int(box_width), int(box_height)])
                    cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (255, 0, 0), -1) #-1은 내부를 색칠하는 것
                    label = f"cell phone: {confidence:.2f}"
                    cv2.putText(image, label, (int(box_x), int(box_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    """
                    time2=time.time() 
                    if time2-time1>5:
                    
                        timer=time.time() 
                        dt_object = datetime.datetime.fromtimestamp(timer)
                        conn = create_connection()
                        data = ('cam1',dt_object, label)
                        insert_data(conn, data)
                        time1= time.time()
                        conn.close()
                    """    
            for p_box in person_boxes:
                for k_box in knife_boxes:
                    if check_overlap(p_box, k_box):
                        cv2.rectangle(image, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (0, 0, 255), 2)
                        cv2.putText(image, 'Armed Person', (p_box[0], p_box[1]-10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
            
            person_boxes = []
            knife_boxes = []
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2)

            cv2.imshow('Detection', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
            pass

"""
try:
  def main():
        timer=time.time() 
        dt_object = datetime.datetime.fromtimestamp(timer)
        conn = create_connection()
        label='cell_phone: 1'
        data = ('cam2',dt_object, label)
        insert_data(conn, data)
        conn.close()
except KeyboardInterrupt:
        pass
   """ 
if __name__ == '__main__':
    main()  
