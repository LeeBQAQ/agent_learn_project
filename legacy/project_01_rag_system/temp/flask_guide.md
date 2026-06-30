# Flask 轻量级 Web 框架

Flask 是一个用 Python 编写的轻量级 Web 应用框架。它被分类为微框架，这意味着它不需要特定的工具或库。它没有数据库抽象层、表单验证，或者其它第三方库提供常用功能的组件。

## 核心特性

### 1. 简洁灵活
Flask 的设计哲学是"微"，核心简单但可扩展：
- 核心只有几个文件
- 不强制使用特定的项目结构
- 可以自由选择数据库、模板引擎等

### 2. 路由系统
Flask 使用装饰器定义路由，非常直观：

```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/user/<username>')
def show_user(username):
    return f'User: {username}'

@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f'Post ID: {post_id}'
```

### 3. 请求和响应
Flask 提供了简单的请求和响应对象：

```python
from flask import request, jsonify

@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    name = data.get('name')
    return jsonify({'message': f'Hello, {name}!'}), 200
```

### 4. 模板引擎
Flask 默认使用 Jinja2 模板引擎：

```python
from flask import render_template

@app.route('/hello/<name>')
def hello(name):
    return render_template('hello.html', name=name)
```

模板文件 `hello.html`：
```html
<!DOCTYPE html>
<html>
<body>
    <h1>Hello, {{ name }}!</h1>
</body>
</html>
```

### 5. 扩展生态系统
Flask 有丰富的扩展库：
- **Flask-SQLAlchemy**: 数据库 ORM
- **Flask-WTF**: 表单验证
- **Flask-Login**: 用户认证
- **Flask-RESTful**: 构建 REST API
- **Flask-CORS**: 跨域支持
- **Flask-JWT**: JWT 认证

### 6. 蓝图（Blueprints）
蓝图用于组织大型应用：

```python
from flask import Blueprint

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login')
def login():
    return 'Login page'

@auth_bp.route('/logout')
def logout():
    return 'Logout page'

# 在主应用中注册蓝图
app.register_blueprint(auth_bp, url_prefix='/auth')
```

## 与 Django 对比

| 特性 | Flask | Django |
|------|-------|--------|
| 类型 | 微框架 | 全功能框架 |
| 学习曲线 | 平缓 | 较陡 |
| 灵活性 | 高 | 中等 |
| 内置功能 | 少 | 多 |
| 适用场景 | 小型项目、API | 大型项目、CMS |
| 社区规模 | 大 | 更大 |

## 快速开始

```bash
pip install flask
```

创建 `app.py`：
```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Welcome to Flask!</h1>'

if __name__ == '__main__':
    app.run(debug=True)
```

运行：
```bash
python app.py
```

Flask 适合小型项目、微服务、REST API 以及需要高度定制化的应用场景。
