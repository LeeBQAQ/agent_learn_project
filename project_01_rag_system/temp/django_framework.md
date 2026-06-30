# Django Web 框架简介

Django 是一个高级的 Python Web 框架，它鼓励快速开发和干净、实用的设计。由经验丰富的开发者构建，Django 处理了 Web 开发的许多麻烦，因此您可以专注于编写应用程序，而无需重新发明轮子。

## 主要特点

### 1. 完整的 MVC 架构
Django 遵循模型-视图-控制器（MVC）设计模式，但在 Django 中被称为 MTV（Model-Template-View）：
- **Model（模型）**：数据访问层，处理与数据库的交互
- **Template（模板）**：表现层，处理显示逻辑
- **View（视图）**：业务逻辑层，决定显示什么数据以及如何显示

### 2. 自带管理后台
Django 最强大的功能之一是自动生成的管理界面。只需几行代码，您就可以获得一个完整的管理后台，用于管理网站内容。

### 3. ORM（对象关系映射）
Django 提供了强大的 ORM 系统，让您可以用 Python 代码而不是 SQL 来操作数据库：

```python
# 定义模型
class Blog(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    pub_date = models.DateTimeField('发布日期')

# 查询数据
blogs = Blog.objects.filter(title__contains='Django').order_by('-pub_date')
```

### 4. URL 路由系统
Django 提供了优雅的 URL 设计系统，支持正则表达式匹配：

```python
from django.urls import path
from . import views

urlpatterns = [
    path('articles/<int:year>/', views.year_archive),
    path('articles/<int:year>/<int:month>/', views.month_archive),
]
```

### 5. 表单处理
Django 提供了强大的表单系统，包括：
- 自动生成 HTML 表单
- 数据验证
- CSRF 保护
- 文件上传处理

### 6. 安全性
Django 内置多种安全机制：
- SQL 注入防护
- 跨站脚本（XSS）防护
- 跨站请求伪造（CSRF）防护
- 点击劫持防护
- SSL/HTTPS 支持

## 适用场景

- 内容管理系统（CMS）
- 社交网络
- 电子商务平台
- 新闻门户
- API 服务后端

## 安装和开始

```bash
pip install django
django-admin startproject mysite
cd mysite
python manage.py runserver
```

Django 适合需要快速开发、安全性要求高、有复杂数据模型的 Web 应用项目。
