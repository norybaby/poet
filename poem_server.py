#coding=utf8
import os
from flask import Flask,request
from write_poem import WritePoem,start_model

app = Flask(__name__)
application = app

path = os.getcwd()    #获取当前工作目录
print(path)

writer = start_model()

# @app.route('/')
# def test(title):
#     return 'test ok'

sytle_help = '<br> para style : 1:自由诗<br> 2:带押韵的自由诗<br> 3:藏头诗<br>4:给定若干字，以最大概率生成诗'
@app.route('/poem')
def write_poem():
    params = request.args
    start_with= ''
    poem_style = 0

    # print(params)
    if 'start' in params :
        start_with = params['start']
    if 'style' in  params:
        poem_style = int(params['style'])

    # return 'hello'
    if  start_with:
         if poem_style == 3:
            return  writer.cangtou(start_with)
         elif poem_style == 4:
            return writer.hide_words(start_with)

    if poem_style == 1:
        return  writer.free_verse()
    elif poem_style == 2:
        return writer.rhyme_verse()

    return 'hello,what do you want? {}'.format(sytle_help)


if __name__ == "__main__":
    app.run()