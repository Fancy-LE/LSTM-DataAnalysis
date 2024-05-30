# 导入工具包
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Input, Embedding, SimpleRNN, LSTM

# from tensorflow.python.keras.preprocessing.text import Tokenizer
# from tensorflow.python.keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer
from keras_preprocessing import sequence
# from tokenizers import Tokenizer
# import sequence

from jieba import lcut

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from tensorflow.python.keras.callbacks import TensorBoard,EarlyStopping
# import PySimpleGUI as sg

# 数据预处理
# """判断一个unicode是否是汉字"""
def is_chinese(uchar):
    if (uchar >= '\u4e00' and uchar <= '\u9fa5') :
        return True
    else:
        return False
def reserve_chinese(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str += i
    return content_str
# 设置去停用词
def getStopWords():
    file = open('./data/stopwords.txt', 'r',encoding='utf8')
    words = [i.strip() for i in file.readlines()]
    file.close()
    return words
# 数据的清洗总体过程
def dataParse(text, stop_words):
    label, content, = text.split('	####	')
    # label = label_map[label]
    content = reserve_chinese(content)
    words = lcut(content)
    words = [i for i in words if not i in stop_words]
    return words, int(label)

# 将上面的过程进行一个整合
def getData(file='./data/hoteldata.txt',):
    file = open(file, 'r',encoding='gbk')
    texts = file.readlines()
    file.close()
    stop_words = getStopWords()
    all_words = []
    all_labels = []
    for text in texts:
        content, label = dataParse(text, stop_words)
        if len(content) <= 0:
            continue
        all_words.append(content)
        all_labels.append(label)
    return all_words,all_labels


## 拿到数据集进行数据集的划分
data,label = getData()
X_train, X_test, train_y, test_y = train_test_split(data,label,test_size=0.2)

# X_train, X_t, train_y, v_y = train_test_split(data,label,test_size=0.3, random_state=42)
# X_val, X_test, val_y, test_y = train_test_split(X_t,v_y,test_size=0.5, random_state=42)

# print(X_train)

## 对数据集的标签数据进行one-hot编码
ohe = OneHotEncoder()
train_y = ohe.fit_transform(np.array(train_y).reshape(-1,1)).toarray()
# val_y = ohe.transform(np.array(val_y).reshape(-1,1)).toarray()
test_y = ohe.transform(np.array(test_y).reshape(-1,1)).toarray()

## 使用keras框架的Tokenizer工具对词组进行编码
## 当我们创建了一个Tokenizer对象后，使用该对象的fit_on_texts()函数，以空格去识别每个词,
## 可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。
max_words = 5000
max_len = 100

tok = Tokenizer(num_words=max_words)  ## 使用的最大词语数为5000
tok.fit_on_texts(data)
# texts_to_sequences 输出的是根据对应关系输出的向量序列，是不定长的，跟句子的长度有关系

train_seq = tok.texts_to_sequences(X_train)
# val_seq = tok.texts_to_sequences(X_val)
test_seq = tok.texts_to_sequences(X_test)

## 将每个序列调整为相同的长度.长度为100，长的切掉，短的补0
train_seq_mat = sequence.pad_sequences(train_seq,maxlen=max_len)
# val_seq_mat = sequence.pad_sequences(val_seq,maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq,maxlen=max_len)
num_classes = 2

## 定义LSTM模型
inputs = Input(name='inputs',shape=[max_len])
## Embedding(词汇表大小,batch大小,每个新闻的词长)
layer = Embedding(max_words+1,128,input_length=max_len)(inputs)
layer = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(layer)
layer = Dense(128,activation="relu",name="FC1")(layer)
layer = Dropout(0.5)(layer)
# 50%的随机丢弃
layer = Dense(2,activation="softmax",name="FC2")(layer)
model = Model(inputs=inputs,outputs=layer)
# model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # 开始！！！！！！
# #模型训练
# ## 当val-loss不再提升时停止训练
# model.fit(train_seq_mat,train_y,batch_size=128,epochs=10,
#                           validation_data=(test_seq_mat,test_y),
#                           callbacks=[TensorBoard(log_dir='./log')]
#                           # callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.0001), TensorBoard(log_dir='./log')]
# )
# # 保存模型
# model.save('model/LSTM.h5')
# del model
# ## 对测试集进行预测
#     # 导入已经训练好的模型
# model = load_model('model/LSTM.h5')
#
# test_pre = model.predict(test_seq_mat)
# pred = np.argmax(test_pre,axis=1)
# real = np.argmax(test_y,axis=1)
# cv_conf = confusion_matrix(real, pred)
# acc = accuracy_score(real, pred)
# precision = precision_score(real, pred, average='micro')
# recall = recall_score(real, pred, average='micro')
# f1 = f1_score(real, pred, average='micro')
# patten = 'test:  acc: %.4f   precision: %.4f   recall: %.4f   f1: %.4f'
# print(patten % (acc,precision,recall,f1,))
# # 画混淆矩阵的图，输出混淆矩阵val_los
# labels = ['negative','active']
# disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels)
# disp.plot(cmap="Blues", values_format='')
# plt.savefig("./static/assets/img/ConfusionMatrix.tif",dpi=400)
# # 结束！！！！！！

# 对可视化界面做准备工作，对某句话的清洗、分词、去停用词，得到一个一维数组
def dataParse_(content, stop_words):
    content = reserve_chinese(content)
    words = lcut(content)
    words = [i for i in words if not i in stop_words]
    return words
def getData_one(text):
    stop_words = getStopWords()
    all_words = []
    content = dataParse_(text, stop_words)
    all_words.append(content)
    return all_words

# 一维数据经过token转换成数字序列的形式
def predict_(text_o):

    data_cut = getData_one(text_o)
    t_seq = tok.texts_to_sequences(data_cut)

    t_seq_mat = sequence.pad_sequences(t_seq, maxlen=max_len)
    model = load_model('model/LSTM.h5')

    t_pre = model.predict(t_seq_mat)
    pred = np.argmax(t_pre, axis=1)
    labels11 = ['negative', 'active']
    pred_lable = []
    for i in pred:
        pred_lable.append(labels11[i])
    return pred_lable[0]
    # return render_template(sentiment_result=result)  # 渲染结果到页面
