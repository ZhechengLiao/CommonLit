# CommonLit Grade Model
Use pretrained bert to do the grade prediction of 3-12 grades students' summary

## Data
Use kaggle competition - commonlit dataset

## Model Architecture
Input -> Bert -> Dropout -> Fully connected layer -> output content, wording grades

## Result
Train Loss: 0.03
Val Loss: 0.49
Training Time: 1h(30 epochs on A100)