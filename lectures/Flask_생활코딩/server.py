from flask import Flask
from flask import request
from flask import redirect
import random

app = Flask(__name__)

nextId = 4
topics = [
    {'id':1, 'title':'html', 'body':'html is ...'},
    {'id':2, 'title':'css', 'body':'css is ...'},
    {'id':3, 'title':'javascript', 'body':'javascript is ...'}
]


def template(contents, content, id=None): # id 파라미터 추가
    contextUI = '' # 디폴트로 아무 내용이 없는 UI
    
    if id != None: # 만약 id 파라미터가 있다면 업데이트 링크 생성
        contextUI = f'''
            <li><a href="/update/{id}/">update</a></li>
            <li><form action="/delete/{id}/" method="POST"><input type="submit" value="delete"></form></li>
            <!-- 템플릿에 델리트 버튼 생성 -->
        '''
    return f'''<!doctype html>
    <html>
        <body>
            <h1><a href="/">WEB</a></h1>
            <ol>
                {contents}
            </ol>
            {content}
            <ul>
                <li><a href = "/create/">create</a></li> 
                {contextUI} <!-- 만약 파라미터 전달되면 update link 생성 -->
            </ul>
            <!-- a href: 링크연결을 하기 위한 부분 -->
            <!-- li: 목록 -->
            <!-- ul: 순서 없는 목록임을 명시 -->
        </body>
    </html>
    '''


# 모두가 동일한 contents를 리턴하는 함수 구현
def getContents():
    liTags = '' # f-string으로 만들 문장
    for topic in topics:
        liTags = liTags + f'<li><a href="/read/{topic["id"]}">{topic["title"]}</a></li>'
    return liTags


@app.route('/')
def index():
    return template(getContents(), '<h2>Welcome</h2>Hello, WEB')


@app.route('/read/<int:id>/')
def read(id): # 여기에서 id로 들어오는 내용은 string임 
              # 따라서 뒤의 == 문법을 이용하기 까다로움 
              # 그러므로 데코레이터에 int라는 hint를 지정해서 정수로 형변환을 해줌
    for topic in topics:
        if id == topic['id']:
            title = topic['title']
            body = topic['body']
    return template(getContents(), f'<h2>{title}</h2>{body}', id)


@app.route('/create/', methods=['GET', 'POST'])
def create():
    if request.method == 'GET':
        content = '''
            <form action="/create/" method="POST">
                <p><input name="title" type="text" placeholder="title"></p>
                <p><textarea name="body" placeholder="body"></textarea></p>
                <p><input type="submit" value="create"></p>
            </form>
        '''
        return template(getContents(), content)
    elif request.method == 'POST':
        global nextId
        title = request.form['title']
        body = request.form['body']
        new_topic = {'id':nextId, 'title':title, 'body':body}
        topics.append(new_topic)
        url = '/read/' + str(nextId) + '/'
        nextId = nextId + 1
        return redirect(url)


@app.route('/update/<int:id>/', methods=['GET', 'POST'])
def update(id):
    if request.method == 'GET':
        title = ''
        body = ''
        for topic in topics:
            if id == topic['id']:
                title = topic['title']
                body = topic['body']
                break
        content = f'''
            <form action="/update/{id}/" method="POST">
                <p><input name="title" type="text" placeholder="title" value="{title}"></p>
                <p><textarea name="body" placeholder="body" value="{body}">{body}</textarea></p>
                <p><input type="submit" value="update"></p>
            </form>
        '''
        return template(getContents(), content)
    elif request.method == 'POST':
        title = request.form['title']
        body = request.form['body']
        for topic in topics:
            if id == topic['id']:
                topic['title'] = title
                topic['body'] = body
                break
        url = '/read/' + str(id) + '/'
        return redirect(url)


@app.route('/delete/<int:id>/', methods=['POST'])
def delete(id):
    for topic in topics:
        if id == topic['id']:
            topics.remove(topic)
            break
    return redirect('/')


app.run(port=5001, debug=True)