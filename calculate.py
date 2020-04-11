import os
import cv2
import matplotlib
# from matplotlib import pyplot as plt


def access_pixels(frame,img):
    height = frame.shape[0]
    weight = frame.shape[1]
    channels = frame.shape[2]
    print("weight : %s, height : %s, channel : %s" %(weight, height, channels))
    intersection=0
    union=0
    for row in range(height):            #遍历高
        for col in range(weight):         #遍历宽
            for c in range(channels):     #便利通道
                pv1 = frame[row, col, c]
                pv2 =img[row,col,c]
                if(pv1<250 and pv2<250):
                    intersection+=1
                    union+=1
                elif(pv1<250 or pv2<250):
                    union+=1

    return intersection,union,intersection/union


if __name__ == "__main__":

    root_path='test_result'

    image_path=os.path.join(root_path,'gt')

    image_paths=os.listdir(image_path)
    
    output_path=os.path.join(root_path,'processed')
    output_paths=os.listdir(output_path)

    print(image_paths[:5])
    print(output_paths[:5])

    precision=0.0
    for input_image,output_image in zip(image_paths,output_paths):

        output_image_path=os.path.join(output_path,output_image)
        input_image_path=os.path.join(image_path,input_image)
     

        target=cv2.imread(output_image_path)
        img=cv2.imread(input_image_path)

        _,_,ratio=access_pixels(target,img)
        precision+=ratio
    print(precision/len(image_paths))

