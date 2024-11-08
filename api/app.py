from flask import Flask

app = Flask(__name__)  # Flaskのインスタンスを作成

# "/"というルート（URL）にアクセスしたときの動作を定義
@app.route("/")
def hello():
    return "Hello, World!"


# APIエンドポイントを作成
@app.route('/predict', methods=['POST'])
def predict():
    # リクエストのJSONデータを取得
    json_data = request.json
    print(json_data)
    
    # "Hello, World!"に続いてJSONデータをレスポンスとして返す
    return jsonify({
        "message": "Hello, World!",
        "data": json_data
    })

if __name__ == "__main__":
    app.run(debug=True)


