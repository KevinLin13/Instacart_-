from flask import Flask, request, url_for
from get_prediction import get_recommendations
import flask
import os

# 創建 Flask 應用程式
app = Flask(__name__, static_folder='images')

# 定義首頁的 route
@app.route('/')
# 當用戶訪問時，會返回 'index.html'
def home():
    """提供首頁模板。"""
    return flask.render_template('index.html')

# 定義 '/predict' 的 route
# predict route 用於處理用戶提交的user_id，這些數據需要被服務器處理（如進行預測），因此使用 POST 請求更合適
@app.route('/predict', methods=['POST']) # 該 route 只接受只接受 POST 請求
def predict():
    to_predict_list = request.form.to_dict() # 儲存了表單提交的數據(user_id)，並轉換為字典
    predictions, time = get_recommendations(to_predict_list) # 利用 get_recommendations 進行預測，會回傳一個字典predictions，跟執行時間。
    print(predictions, time)

    if 'recommend' not in predictions.keys(): # 表示是新用戶
        image_paths = {}
        for _, value in predictions['top'].items():
            folder_name = value
            jpg_path = os.path.join(app.static_folder, folder_name.replace(' ', '_'), f"{folder_name}_1.jpg")
            png_path = os.path.join(app.static_folder, folder_name.replace(' ', '_'), f"{folder_name}_1.png")
            jpeg_path = os.path.join(app.static_folder, folder_name.replace(' ', '_'), f"{folder_name}_1.jpeg")

            if os.path.exists(jpg_path):
                image_paths[value] = url_for('static', filename=f"{folder_name.replace(' ', '_')}/{folder_name}_1.jpg")
            elif os.path.exists(png_path):
                image_paths[value] = url_for('static', filename=f"{folder_name.replace(' ', '_')}/{folder_name}_1.png")
            elif os.path.exists(jpeg_path):
                image_paths[value] = url_for('static', filename=f"{folder_name.replace(' ', '_')}/{folder_name}_1.jpeg")
            else:
                image_paths[value] = None
        return flask.render_template("new_user_recommendation.html", predictions=predictions, image_paths=image_paths) # 生成 new_user_recommendation.html
    else:
        image_paths = {}
        for _, value in predictions['recommend'].items():
            folder_name = value
            jpg_path = os.path.join(app.static_folder, folder_name.replace(' ', '_'), f"{folder_name}_1.jpg")
            png_path = os.path.join(app.static_folder, folder_name.replace(' ', '_'), f"{folder_name}_1.png")
            jpeg_path = os.path.join(app.static_folder, folder_name.replace(' ', '_'), f"{folder_name}_1.jpeg")

            if os.path.exists(jpg_path):
                image_paths[value] = url_for('static', filename=f"{folder_name.replace(' ', '_')}/{folder_name}_1.jpg")
            elif os.path.exists(png_path):
                image_paths[value] = url_for('static', filename=f"{folder_name.replace(' ', '_')}/{folder_name}_1.png")
            elif os.path.exists(jpeg_path):
                image_paths[value] = url_for('static', filename=f"{folder_name.replace(' ', '_')}/{folder_name}_1.jpeg")
            else:
                image_paths[value] = None
        return flask.render_template("predict.html", predictions=predictions, image_paths=image_paths) # 舊用戶生成 predict.html

# 啟動應用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
