from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import os
import time
import sklearn

app = FastAPI()

# CORS設定を更新
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのオリジンを許可に戻す
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルを読み込み
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', '4s_model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model type: {type(model)}")
    print(f"Model version: {joblib.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    
    # シンプルなテスト予測
    test_features = np.zeros(80)  # 80次元のテスト特徴量
    test_prediction = model.predict(test_features.reshape(1, -1))
    print(f"Test prediction successful: {test_prediction}")
except Exception as e:
    print(f"Error loading or testing model: {str(e)}")
    raise e

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

@app.get("/")
async def root():
    try:
        # モデルが読み込めるか確認
        if os.path.exists(MODEL_PATH):
            return JSONResponse(
                content={
                    "status": "running",
                    "message": "WebSocket server is running",
                    "model_loaded": True
                }
            )
        else:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Model file not found",
                    "model_path": MODEL_PATH
                },
                status_code=500
            )
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e)
            },
            status_code=500
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_host = websocket.client.host
    print(f"New WebSocket connection attempt from {client_host}")
    await websocket.accept()
    print(f"WebSocket connection accepted from {client_host}")
    
    try:
        while True:
            try:
                data = await websocket.receive_json()
                print(f"Received data from {client_host} at {time.strftime('%H:%M:%S')}")
                print(f"Data shape: {len(data['eegData'])} bands, {len(data['eegData'][0])} samples")
                
                try:
                    features = extract_features(data['eegData'])
                    print(f"Extracted features shape: {len(features)}")
                    prediction = int(model.predict([features])[0])
                    print(f"Prediction successful: {prediction}")
                    
                    response = {"prediction": prediction}
                    print(f"Sending response: {response}")
                    await websocket.send_json(response)
                    print("Response sent successfully")
                    
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
                    print(f"Features: {features[:10]}...")  # 最初の10個の特徴量を表示
                    continue
                
            except WebSocketDisconnect:
                print(f"Client {client_host} disconnected")
                break
            except Exception as e:
                print(f"Error processing data: {str(e)}")
                print(f"Raw data: {data}")  # 受信データの内容を表示
                continue
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        print(f"WebSocket connection closed") 