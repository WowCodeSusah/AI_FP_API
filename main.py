from fastapi import FastAPI, File, UploadFile
import keras
import pose
import pandas as pd

app = FastAPI()

@app.get('/')
async def index():
    return {"Pose Data Estimation": "This is Index"}

@app.get('/test')
async def test():
    return {"Test": "Succesful"}

@app.post('/predictCNN/')
async def predict(files: list[UploadFile]=File(...)):
    print('yes')
    Model = keras.models.load_model('models/modelCNN.keras')
    try: 
        for file in files:
            file_path = f"C:\\Users\\micha\\Downloads\\AI FP\\API\\images\\{file.filename}"
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            
            pose_dict = pose.run('images/' + file.filename)

            if len(pose_dict) == 99:
                df = pd.DataFrame(pose_dict, index=[0])
                numpy_array = df.to_numpy()

                test = Model.predict(numpy_array)

                return {"Accuracy": "{:.4f}".format(test.tolist()[0][0])}
            else:
                return {"Message": "Pose data was incorrectly estimated"}
        
    except Exception as e:
        return {"message": e.args}

    Model = keras.models.load_model('models/modelCNN.keras')
    try: 
        for file in files:
            file_path = f"C:\\Users\\micha\\Downloads\\AI FP\\API\\images\\{file.filename}"
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            
            pose_dict = pose.run('images/' + file.filename)

            if len(pose_dict) == 99:
                df = pd.DataFrame(pose_dict, index=[0])
                numpy_array = df.to_numpy()

                test = Model.predict(numpy_array)

                return {"Accuracy": test.tolist()[0][0], "PoseData": pose_dict}
            else:
                return {"Message": "Pose data was incorrectly estimated"}
        
    except Exception as e:
        return {"message": e.args}

@app.post('/predictCNNwithATT/')
async def predict(files: list[UploadFile]):
    Model = keras.models.load_model('models/modelCNNwithATT.keras')
    try: 
        for file in files:
            file_path = f"C:\\Users\\micha\\Downloads\\AI FP\\API\\images\\{file.filename}"
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            
            pose_dict = pose.run('images/' + file.filename)

            if len(pose_dict) == 99:
                df = pd.DataFrame(pose_dict, index=[0])
                numpy_array = df.to_numpy()

                test = Model.predict(numpy_array)

                return {"Accuracy": test.tolist()[0][0], "PoseData": pose_dict}
            else:
                return {"Message": "Pose data was incorrectly estimated"}
        
    except Exception as e:
        return {"message": e.args}

@app.post('/predictVGG/')
async def predict(files: list[UploadFile]):
    Model = keras.models.load_model('models/modelVGG.keras')
    try: 
        for file in files:
            file_path = f"C:\\Users\\micha\\Downloads\\AI FP\\API\\images\\{file.filename}"
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            
            pose_dict = pose.run('images/' + file.filename)

            if len(pose_dict) == 99:
                df = pd.DataFrame(pose_dict, index=[0])
                numpy_array = df.to_numpy()

                test = Model.predict(numpy_array)

                return {"Accuracy": test.tolist()[0][0]}
            else:
                return {"Message": "Pose data was incorrectly estimated"}
        
    except Exception as e:
        return {"message": e.args}