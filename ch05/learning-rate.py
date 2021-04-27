import torch
from torch.autograd import Variable


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


x_train = [8, 8.5, 7.5]
y_train = [8.5, 9.5, 7.5]

inputDim = 1  # takes variable 'x'
outputDim = 1  # takes variable 'y'
learningRate = 0.1
epochs = 100

model = linearRegression(inputDim, outputDim)
##### For GPU #######
if torch.cuda.is_available():
    model.cuda()
# the loss (Mean Squared Error)
criterion = torch.nn.MSELoss()
# optimization (Stochastic Gradient Descent) functions
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

for epoch in range(epochs):
    labels = y_train
    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()
    # get output from the model, given the inputs
    outputs = model(x_train)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    print(loss)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))
