import os
import cv2
import matplotlib
# from matplotlib import pyplot as plt


def access_pixels(frame,img,output_image):
    print(frame.shape)  #shape内包含三个元素：按顺序为高、宽、通道数
    height = frame.shape[0]
    weight = frame.shape[1]
    channels = frame.shape[2]
    print("weight : %s, height : %s, channel : %s" %(weight, height, channels))
    
    for row in range(height):            #遍历高
        for col in range(weight):         #遍历宽
            for c in range(channels):     #便利通道
                pv = frame[row, col, c]
                if(pv>44):
                    img[row, col, c] = 0     #全部像素取反，实现一个反向效果
    # plt.imshow(img)
    # plt.show()
    cv2.imwrite(output_image,img)


if __name__ == "__main__":
    input_image='0002_1_input.png'
    image='0002_3_defocus_map_norm_out.png'
    output=input_image.split('.')[0]+'_focus'+'.png'
    root_path='author_dir'

    image_path=os.path.join(root_path,'image')
    out_norm_path=os.path.join(root_path,'out_norm')

    image_paths=os.listdir(image_path)
    out_norm_paths=os.listdir(out_norm_path)
    print(image_paths[:5])
    print(out_norm_paths[:5])
    output_path=os.path.join(root_path,'processed')
    os.makedirs(output_path,exist_ok=True)
    for input_image,out_norm in zip(image_paths,out_norm_paths):

        norm_image_path=os.path.join(out_norm_path,out_norm)
        input_image_path=os.path.join(image_path,input_image)
        output=input_image.split('.')[0]+'_focus'+'.png'
        output_image=os.path.join(output_path,output)

        print(norm_image_path)
        print(input_image_path)
        target=cv2.imread(norm_image_path)
        img=cv2.imread(input_image_path)

        access_pixels(target,img,output_image)

