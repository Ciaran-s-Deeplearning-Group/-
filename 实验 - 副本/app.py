import numpy as np  
from flask import Flask, request, jsonify, render_template  
import pickle  
  
app = Flask(__name__)  
  
# 加载模型和缩放器  
with open("scaler_stroke.pkl", "rb") as f:  
    scaler = pickle.load(f)  
  
with open("stroke.pkl", "rb") as f:  
    logreg = pickle.load(f)  
  
# 假设模型训练时使用了这些特征名称（这些应该与前端表单中的字段匹配）  
# feature_names = ["age", "bp", "sugar", "heart_disease", ...]  # 根据实际情况填写  
  
def predict_stroke_from_features(features):  
    try:  
        # 确保特征数量正确，并转换为 numpy 数组  
        if len(features) != scaler.n_features_in_:  
            raise ValueError("Number of features does not match the model's expectations.")  
          
        final_features = np.array(features).reshape(1, -1)  # 重塑为 (1, n_features)  
        scaled_features = scaler.transform(final_features)  
        prediction = logreg.predict(scaled_features)  
          
        if prediction == 1:  
            prediction_text = 'Oops SORRY! You have stroke!'  
        else:  
            prediction_text = 'Great News! You are healthy!'  
          
        return prediction_text  
    except Exception as e:  
        return str(e)  # 返回错误消息  
  
@app.route('/')  
def home():  
    return render_template('index_main.html')  
  
@app.route("/stroke")  
def stroke():  
    return render_template("stroke.html")  
  
@app.route('/_canpredictcer', methods=['POST'])  
def predict_stroke():  
    # 从表单中获取特征值，注意这里需要确保表单字段与模型特征匹配  
    # 假设表单字段名称与 feature_names 列表中的名称相同  
    features = []  
    for name in feature_names:  # 确保这里的 name 与表单中的 input name 对应  
        value = request.form.get(name)  
        try:  
            # 根据实际情况转换数据类型，可能是 int、float 或保持为 str（如果模型需要）  
            # 这里假设所有特征都是浮点数  
            features.append(float(value))  
        except ValueError:  
            # 如果转换失败，可以添加错误处理或默认值  
            features.append(np.nan)  # 或者使用其他默认值，但模型可能需要处理 NaN  
      
    prediction_text = predict_stroke_from_features(features)  
    return render_template('stroke.html', prediction_text=prediction_text)  
  
if __name__ == "__main__":  
    app.run(debug=True, host='0.0.0.0', port=5000)  # 默认端口是5000，但您可以根据需要更改它