# This code is to make the streamlit app.

import os

path = 'polygonus/MathMax/MNIST'
cwd = os.getcwd()


import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from IPython.display import display


# First lets construct the models.

class CNN(nn.Module):
  def __init__(self, num_classes):
    super(CNN, self).__init__()
    # Initial image size is 28x28.
    # Cambio 3. Agregar tres capas de convoluciones con padding. 
    self.conv1 = nn.Conv2d(1, 16, 3, padding = 'same')  
    self.conv1_1 = nn.Conv2d(16,16,3, padding = 'same')
    self.conv1_2 = nn.Conv2d(16,16,3, padding = 'same')
    self.padd1 = nn.MaxPool2d(kernel_size = 2) # Size changed to 14 x 14.
    self.conv2 = nn.Conv2d(16,32,3, padding = 'same') 
    self.conv2_1 = nn.Conv2d(32,32,3, padding = 'same')
    self.conv2_2 = nn.Conv2d(32,32,3, padding = 'same')
    self.padd2 = nn.MaxPool2d(kernel_size = 2) # Size changed to 7 x 7.
    self.conv3 = nn.Conv2d(32,64,3, padding = 'same') 
    self.conv3_1 = nn.Conv2d(64,64,3, padding = 'same')
    self.conv3_2 = nn.Conv2d(64,64,3, padding = 'same')
    self.conv4 = nn.Conv2d(64,32,1)
    self.conv5 = nn.Conv2d(32,1,1) # Size unchanged.  
    self.lin1 = nn.Linear(7**2, num_classes)
    self.initialize_weights()

  def forward(self, x):
    # Softmax not applied since it will be applied later on the loss function. 
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv1_1(out))
    out = F.relu(self.conv1_2(out))
    out = self.padd1(out)
    out = F.relu(self.conv2(out))
    out = F.relu(self.conv2_1(out))
    out = F.relu(self.conv2_2(out))
    out = self.padd1(out)
    out = F.relu(self.conv3(out))
    out = F.relu(self.conv3_1(out))
    out = F.relu(self.conv3_2(out))
    out = F.relu(self.conv4(out))
    out = F.relu(self.conv5(out))
    out = out.view(-1,7**2)
    return self.lin1(out)

  def initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)

        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

@st.cache(suppress_st_warning = True)
def make_model():
  device = torch.device('cpu')
  cnn_model = CNN(num_classes = 10)
  cnn_model.to(device)
  cnn_model.load_state_dict(torch.load('MNIST_model.pth', map_location = device))
  cnn_model.eval()
  return cnn_model

cnn_model = make_model()


def preprocessing(img, factor = 1.6, factor2 = 4.5):
  matrix = np.asarray(img)
  mu = np.mean(matrix)
  delta = np.std(matrix)
  lim_inf = mu - factor*delta
  lim_sup = mu + factor*delta
  mask = np.where((matrix < lim_inf)|(lim_sup < matrix), matrix, 0)
  rows, cols = np.where(mask > 0)
  row_mu = np.mean(rows)
  row_delta = np.std(rows)
  inf_row = max(0,int(row_mu - factor2*row_delta))
  sup_row = min(mask.shape[0]-1,int(row_mu + factor2*row_delta))
  col_mu = np.mean(cols)
  col_delta = np.mean(delta)
  inf_col = max(0,int(col_mu - factor2*col_delta))
  sup_col = min(mask.shape[1]-1,int(col_mu + factor2*col_delta))
  mask = mask[inf_row:sup_row,inf_col:sup_col]
  return Image.fromarray(mask)

# defining the sections of the website. 


 
header = st.container()
interactive = st.container()
datasets = st.container()
the_model = st.container()

with header:
    st.title('Digit Classification')
    st.text('Built using MNIST dataset.')

with interactive:
    st.header('Digit Classifier')
    st.text('Draw a digit in the canvas below. \n'
            'Press submit when ready.'
            )
    img_input = st_canvas(width = 300, height = 300,
                          fill_color ="rgba(255, 0, 0, 1)")
    switch = st.button('Submit.')
    matrix = img_input.image_data[:,:,3]
    values = len(list(set(list(matrix.reshape(-1)))))
    if values > 1 and switch:
      num_image = Image.fromarray(np.uint8(matrix))
      num_image = preprocessing(num_image)
      num_image = num_image.resize((28,28))
      Input = torch.from_numpy(np.asarray(num_image).reshape(1,1,28,28)).float()
      Outcome = cnn_model(Input)
      _, pred = torch.max(Outcome, 1)
      st.subheader(f'Model result: {pred.item()}')
    else:
      st.subheader('Model result: None.')

with datasets:
  st.header('The Training dataset')
  st.text('This model was trainned with the public data-\n'
            'set MNIST.Made by Institute of Standars and\n'
            'Technology of USA. It contains 60 thousand\n'
            'examples of handwritten digits for trainning\n'
            'and 10 thousand more for testing.'
          )
  st.subheader('Sample of the digits of MNIST.')
  examples = Image.open('MnistExamples.png')
  w, h = examples.size
  w = 8*(w//10)
  h = 8*(h//10)
  st.image(examples.resize((w,h)))
  st.text('Taken from the Wikipedia article.\n'
          'Made by Josef Steppan.')

with the_model:
  st.header('Our Model')
  st.text('The model used is a convolutional neural\n'
          'network having 122,901 parameters. It has\n'
          'a validation accuracy of 99.16%.'
          )