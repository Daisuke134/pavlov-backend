from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import os
import time

app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルを4s_model.pklに戻す
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', '4s_model.pkl')
model = joblib.load(MODEL_PATH)

# 特徴量抽出関数を4電極用に戻す
def extract_features(data):
    features = []
    for wave_data in data:  # 各周波数帯（delta, theta, alpha, beta, gamma）
        for electrode in range(4):  # 4電極それぞれについて
            electrode_data = [wave_data[i] for i in range(electrode, len(wave_data), 4)]
            # 40サンプル（4秒 × 10Hz）から4つの特徴量を計算
            mean = np.mean(electrode_data)
            std = np.std(electrode_data)
            max_val = np.max(electrode_data)
            min_val = np.min(electrode_data)
            features.extend([mean, std, max_val, min_val])
    return features  # 80特徴量（5周波数帯 × 4電極 × 4特徴量）

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("New WebSocket connection")
    await websocket.accept()
    
    while True:
        try:
            data = await websocket.receive_json()
            start_time = time.time()
            
            # データ形状を確認
            print(f"Received data at {time.strftime('%H:%M:%S')}:")
            print(f"Samples per band: {len(data['eegData'][0])}")
            print(f"Expected: 40 samples (4秒 × 10Hz)")
            
            # 特徴量抽出と予測を実行
            features = extract_features(data['eegData'])
            prediction = model.predict([features])[0]
            
            # 処理時間を計測
            process_time = time.time() - start_time
            print(f"Processing time: {process_time*1000:.2f}ms")
            print(f"Prediction result: {prediction}")
            
            await websocket.send_json({"prediction": int(prediction)})
            
        except Exception as e:
            print(f"Error in WebSocket connection: {e}")
            break 