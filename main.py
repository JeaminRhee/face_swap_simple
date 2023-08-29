import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


st.set_page_config(page_title='얼굴교체',  layout = 'wide', initial_sidebar_state = 'auto')


# set title
st.title('얼굴 교체')

# set header
st.header('Please upload an image that you want to be - The image must feature a single face.')

# upload want_to_file (너가 되고 싶은 이미지)
want_to_file = st.file_uploader('', type=['jpeg', 'jpg', 'png'], key="1")


# set header
st.header('Please upload your face image - The image must feature a single face.')

# upload source_file (너의 얼굴)
source_file = st.file_uploader('', type=['jpeg', 'jpg', 'png'], key="2")


""" INSIGHT FACE - #01. DETECT FACES """ 
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640,640))


source_img = ins_get_image(source_file)
want_to_img = ins_get_image(want_to_file)


source_face = app.get(source_img)
faces = app.get(want_to_img)

""" INSIGHT FACE - #02. FACE SWAPPING """
swapper = insightface.model_zoo.get_model('inswapper_128.onnx',
                                          download=False,
                                          download_zip=False)

res = want_to_img.copy()
res = swapper.get(res, faces[0], source_face, paste_back=True)

""" SHOW SWAPPED FACE """
st.image(res[:, :, ::-1], caption="바뀐 당신의 얼굴")
plt.show()


# load classifier
# model = load_model('./model/pneumonia_classifier.h5')

# load class names
# with open('./model/labels.txt', 'r') as f:
#     class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
#    f.close()

# display image
"""
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
"""
