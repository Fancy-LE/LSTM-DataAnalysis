from flask import Flask, jsonify, request
from flask import render_template
import sqlite3
import model

app = Flask(__name__)


@app.route('/')
def home():  # put application's code here
    return render_template("index.html")

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/emotion')
def emotion():
    return render_template("emotion.html")

@app.route('/attraction')
def attraction():
    return render_template('attraction.html')  # 前端返回comments

@app.route('/get_data')
def get_data():
    name = []  # 景点名称score
    num = []  # 每个景点的评论数
    con = sqlite3.connect("hot.db")
    cur = con.cursor()
    sql = 'select * from hot '
    data = cur.execute(sql)
    for item in data:
        name.append(str(item[0]))
        num.append(item[1])
    cur.close()
    con.close()
    return jsonify({'name':name,'num':num})

@app.route('/get_month')
def get_month():
    num = [] # 每个月份的评论数
    con = sqlite3.connect("month.db")
    cur = con .cursor()
    sql = 'select 评论数 from month'
    data = cur.execute(sql)
    for item in data:
        num.append(item[0])
    cur.close()
    con.close()
    return jsonify({'num':num})

@app.route('/time')
def time():
    return render_template("time.html")

@app.route('/ciyun')
def ciyun():
    datalist = []
    con = sqlite3.connect("comments.db")
    cur = con.cursor()  # 建立游标
    sql = 'select * from comments where LENGTH(评论内容)<60 order by random() limit 20'
    data = cur.execute(sql)
    for item in data:
        datalist.append(item)
    cur.close()
    con.close()
    return render_template("ciyun.html",comments = datalist)

# 定义一个预测路由，以post方式提交数据
@app.route('/predict',methods = ['POST'])
def predict():
    text = request.form.get('text') #从表单获取文本输入
    result = model.predict_(text) # 使用模型进行预测
    print(result)
    return jsonify({'result': result}) # 将预测结果作为JSON返回


if __name__ == '__main__':
    app.debug = True
    app.run()
