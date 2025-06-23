from example_pb2_grpc import PredictionServiceServicer
import example_pb2_grpc
import example_pb2 as msgs
import grpc
import os
from concurrent import futures
import joblib
import pandas as pd

model_path = os.path.join("..", "svm_iris_model.pkl")
model = joblib.load(model_path)

class PredictionService(PredictionServiceServicer):
    def GetPrediction(self,request,context):
        features = [[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]]
        df = pd.DataFrame(features, columns=[
            'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
        ])
        prediction = model.predict(df)[0]
        prediction2 = msgs.Prediction()
        prediction2.pred = prediction
        return prediction2

class PredictionService2(PredictionServiceServicer):
    def GetPrediction(self,request,context):
        prediction = msgs.Prediction()
        prediction.pred = 96.9
        return prediction

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    example_pb2_grpc.add_PredictionServiceServicer_to_server(PredictionService(),server)
    server.add_insecure_port("localhost:5001")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()