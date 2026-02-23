# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The given neural network architecture consists of an input layer with one neuron two hidden layers with three neurons each and an output layer with one neuron. This structure suggests that the model is designed for a regression or binary classification task, where a single input feature is processed through multiple layers of transformations to produce a single output value. The fully connected layers indicate that each neuron in one layer is connected to all neurons in the next, allowing the network to learn complex relationships within the data. The problem statement for this model could be predicting a continuous output or classifying an input into one of two categories. The hidden layers enable the model to capture non-linear patterns in the data, making it suitable for problems where simple linear models are insufficient.

## Neural Network Model

<img width="773" height="527" alt="image" src="https://github.com/user-attachments/assets/51723f55-32c2-439d-b5b4-7a8298fc9a24" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: GAYATHRI S
### Register Number: 212224230073
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}
  def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
GAYU_brain=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(GAYU_brain.parameters(), lr=0.001)

def train_model(GAYU_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(GAYU_brain(X_train), y_train)
        loss.backward()
        optimizer.step()
        GAYU_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information
<img width="735" height="283" alt="image" src="https://github.com/user-attachments/assets/f3557d71-880a-47e2-b4c7-b45243806912" />



## OUTPUT

<img width="811" height="573" alt="image" src="https://github.com/user-attachments/assets/f26d1efd-2da2-4482-9a7d-f32872b5fdf7" />



### New Sample Data Prediction


<img width="811" height="155" alt="image" src="https://github.com/user-attachments/assets/b81a8261-8805-423d-b0a6-8a6b46d932a0" />



## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
