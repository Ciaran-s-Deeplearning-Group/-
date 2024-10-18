import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score  
import joblib  # 确保导入了joblib  
import os  
from pathlib import Path  
# 加载数据，假设CSV文件的第一列是'id'，我们不需要它作为特征  
df = pd.read_csv('C:\\Program Files (x86)\\新建文件夹\\新建文件夹\\实验 - 副本\\Datasets\\stroke.csv')  
# 由于id列通常是唯一标识符，我们不需要它作为特征，所以不在features列表中包含它  
features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']  
target = 'stroke'  
  
# 对分类变量进行编码  
label_encoders = {}  
for feature in features:  
    if df[feature].dtype == type(object):  
        le = LabelEncoder()  
        df[feature] = le.fit_transform(df[feature])  
        label_encoders[feature] = le  
  
# 将特征和目标变量分开  
X = df[features]  
y = df[target]  
  
# 标准化特征  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  
  
# 将数据集分为训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  
  
# 选择并训练模型  
model = RandomForestClassifier(n_estimators=100, random_state=42)  
model.fit(X_train, y_train)  
  
# 在测试集上进行预测  
y_pred = model.predict(X_test)  
  
# 计算并打印准确率  
accuracy = accuracy_score(y_test, y_pred)  
print(f'Model accuracy: {accuracy:.2f}')  
  
# 创建一个用于保存模型的文件夹（如果不存在）  
model_dir = Path.home() / 'models'  
os.makedirs(model_dir, exist_ok=True)  
  
# 设置模型文件的路径  
joblib_file = model_dir / 'random_forest_model.pkl'  
  
# 保存模型  
joblib.dump(model, joblib_file)