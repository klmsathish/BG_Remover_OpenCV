import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import cv2 as cv
# from matplotlib import pyplot as plt
import sys
import io
import base64

st.markdown("<h1 style='text-align: center;margin-top: -80px; color: blue;'>Automatic Background Remover</h1>", unsafe_allow_html=True)
st.write("This is a image background remover web app using python")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
if file is None:
    st.text("You haven't uploaded an image file")
else:
    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    # image = Image.open(file)
    try:
        file_bytes = np.asarray(bytearray(file.read()),dtype=np.uint8)
        img = cv.imdecode(file_bytes, 1)
        original = img.copy()

        low_thresh = 7
        high_thresh = 7
        edges = cv.GaussianBlur(img, (21, 51), 3) #In Gaussian Blur operation, the image is convolved with a Gaussian filter instead of the box filter. The Gaussian filter is a low-pass filter that removes the high-frequency components are reduced.
        edges = cv.cvtColor(edges, cv.COLOR_BGR2GRAY) # Changing colour image to B/W
        edges = cv.Canny(edges,low_thresh, high_thresh) #Canny Edge Detection is a popular edge detection algorithm.It helps in Noise Reduction,Finding Intensity Gradient of the Image
        #In Binary Thresholding each pixel value is compared with the threshold value. If the pixel value is smaller than the threshold, it is set to 0, otherwise, it is set to a maximum value (generally 255)
        #In Otsu Thresholding, a value of the threshold isn’t chosen but is determined automatically. A bimodal image (two distinct image values) is considered. The histogram generated contains two peaks. So, a generic condition would be to choose a threshold value that lies in the middle of both the histogram peak values.
        _, thresh = cv.threshold(edges, 0, 255, cv.THRESH_BINARY  + cv.THRESH_OTSU)

        #The value of each pixel in the output image is based on a comparison of the corresponding pixel in the input image with its neighbors.
        #By choosing the size and shape of the kernel, you can construct a morphological operation that is sensitive to specific shapes regarding the input image.
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

        #Closing is useful in closing small holes inside the foreground objects, or small black points on the object.
        mask = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=4)
        data = mask.tolist()
        sys.setrecursionlimit(10**8)
        for i in  range(len(data)):
            for j in  range(len(data[i])):
                if data[i][j] !=  255:
                    data[i][j] =  -1
                else:
                    break
            for j in  range(len(data[i])-1, -1, -1):
                if data[i][j] !=  255:
                    data[i][j] =  -1
                else:
                    break
        image = np.array(data)
        image[image !=  -1] =  255
        image[image ==  -1] =  0
        mask = np.array(image, np.uint8)
        #cv.bitwise_and - Calculates the per-element bit-wise conjunction of two arrays or an array and a scalar.
        result = cv.bitwise_and(original, original, mask=mask)
        result[mask ==  0] =  255
        cv.imwrite('bg.png', result)
        img = Image.open('bg.png')
        img.convert("RGBA")
        datas = img.getdata()
        newData = []
        for item in datas:
            if item[0] ==  255  and item[1] ==  255  and item[2] ==  255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        img.putdata(newData)
        st.markdown('<p class="big-font">After removing Background</p>', unsafe_allow_html=True)
        st.image(img,width = 400,height = 180)
    finally:
            print("done")
