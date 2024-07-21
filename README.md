# Face Recognition for Supermarket Customers

## Project Overview

This project implements a face recognition system for supermarket customers, linking each recognized face to an SQLite database that stores their purchase history and profile information. The system includes image capture, model training, evaluation, and integration with a database to provide a complete solution.

### Project Description
This project is conducted as a final exam project for Artificial Intelligence for Data Science. It implements a face recognition system using Python and OpenCV, demonstrating the practical application of AI techniques in real-world scenarios.
If you need more explanation about how to do this, please check this medium article below :
- [Step 1](https://medium.com/@986110101/pengenalan-wajah-1-6dc7d788fd07)
- [Step 2](https://medium.com/@986110101/pengenalan-wajah-2-5152fa2ee5da)
- [Step 3](https://medium.com/@986110101/pengenalan-wajah-3-c05a6422113e)
- [Step 4](https://medium.com/@986110101/pengenalan-wajah-4-a00b3213e49d)
- [Step 5](https://medium.com/@986110101/pengenalan-wajah-5-cb65f3726e44)

## Files Description

### `capture.py`

- **Purpose**: Captures images from the camera and labels them with customer profile information.
- **Usage**: Used for collecting training data for the face recognition model.

### `model.py`

- **Purpose**: Trains the face recognition model using the data collected from `capture.py`.
- **Output**: Generates `training.xml`, which contains the trained model.

### `sql.py`

- **Purpose**: Connects to the camera to detect faces and provides labels according to the model. Also integrates with the SQLite database to display customer information associated with the detected face.
- **Usage**: Demonstrates real-time face recognition and database integration.

### `eval.py`

- **Purpose**: Evaluates the performance of the trained model using metrics such as F1-score, Precision, and Recall.
- **Usage**: Provides insights into the effectiveness of the face recognition system.

### `finish.py`

- **Purpose**: Final implementation where detected faces are labeled with customized settings (font, size, color, etc.) and displayed on the camera feed.
- **Usage**: Represents the polished version of the face recognition system.

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nnovallll/simple_face_detection.git
   cd simple_face_detection
   ```

2. **Install dependencies**:
   Make sure you have Python installed. Install the required libraries using pip:
   ```bash
   pip install opencv-python dlib sqlite3
   ```

3. **Set up the SQLite database**:
   Make sure to configure your SQLite database according to your needs. The `sql.py` file assumes a pre-existing database schema.

## Usage

### 1. Data Collection
- Run `capture.py` to capture images and label them.
- Ensure you have a variety of images for diverse representation.
- For face detection, use haarcascades from [this GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).
- Prepare 5 customer faces. Feel free to explore the faces you want to use for this project.
- Ensure images are properly labeled and saved in the `/DataSet` folder.
- Make sure the images meet the criteria you've set, so the model can be built effectively later.

### 2. Database Integration
- Set up a database to store information about recognized individuals, for this project i recomended to used SQLite.
- The database should include fields such as:
    - Name
    - Gender
    - Last visit date
    - Estimated age
    - Top 3 purchased products
    - Suggested products for next purchase
- Ensure proper connection between the face recognition system and the database for real-time information retrieval.

### 3. Model Training
- Execute `model.py` to train the face recognition model.
- This script will process the images in the `/DataSet` folder.
- The training process will generate a `training.xml` file, which will be used for face recognition.
- Ensure you have sufficient computational resources, as training may take some time depending on the dataset size.

### 4. Face Recognition
- Run `sql.py` to test the face detection and recognition capabilities.
- This script will use the trained model to recognize faces in real-time or from stored images.
- Recognized faces will be displayed with their corresponding labels.
- Additional information from the database will be shown for each recognized individual.

### 5. Model Evaluation
- Use `eval.py` to assess the performance of your trained model.
- This script will run the model against a test set and calculate various evaluation metrics.
- Key metrics such as accuracy, precision, recall, and F1-score will be reported.
- Review these metrics carefully to understand the model's strengths and limitations.

### 6. System Implementation
- Execute `finish.py` for the complete implementation of the face recognition system.
- This script combines all the previous steps into a cohesive application.
- Detected faces will be displayed in real-time with customized labels.
- The system will integrate with the database to provide relevant information for each recognized face.
- Ensure all dependencies are correctly installed and paths are properly set before running this script.


## Documentation

### **Data Collection**

- Process of capture data

 ![image](https://github.com/user-attachments/assets/ccef7d6c-b9e0-4030-b742-2535b9afcf7b)

- Result

 ![image](https://github.com/user-attachments/assets/7fb3b13c-e2f6-4aaf-8a67-df1e5d73a2cf)

## **Database Integration**

- Create Database

![image](https://github.com/user-attachments/assets/27ec0c1b-b095-4fd9-b0cf-3717bbcd1641)

### **Model Training**

- Process of model training

  ![image](https://github.com/user-attachments/assets/61dcd919-4d87-4d65-a752-fa9bbd4c71d9)

- Result

  ![image](https://github.com/user-attachments/assets/c3485cdf-8054-458f-a162-beed5e014f60)
  
### **Model Evaluation**

- Model evaluation with f1-score, recall, and precision

![image](https://github.com/user-attachments/assets/9c3fab52-6818-4417-aa44-76402aaee48a)

### System Integration 

- Checking the connection between model and database

![image](https://github.com/user-attachments/assets/8251e23e-da56-491d-9231-8af8efb6d3f0)

- Finish system

![image](https://github.com/user-attachments/assets/f546f30d-6309-4795-80e4-2af8b1ee2962)

## Contributing

Feel free to open issues or submit pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
