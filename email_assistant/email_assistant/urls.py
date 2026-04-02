from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse

def home(request):
    return HttpResponse("""
    <html>
    <head>
        <title>Email Assistant</title>
        <style>
            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: #0f172a;
                color: #ffffff;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                text-align: center;
            }
            h1 {
                font-size: 36px;
                margin-bottom: 10px;
            }
            p {
                font-size: 18px;
                color: #cbd5f5;
            }
            .tag {
                margin-top: 20px;
                padding: 8px 16px;
                background: #1e293b;
                border-radius: 20px;
                display: inline-block;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to Email Assistant Backend API</h1>
            <p>Your backend server is running successfully.</p>
            <p>Developed by Muhammad Umar Tariq with love 😀</p>
            <div class="tag">Django API Server</div>
        </div>
    </body>
    </html>
    """)

urlpatterns = [
    path("", home),
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),
]