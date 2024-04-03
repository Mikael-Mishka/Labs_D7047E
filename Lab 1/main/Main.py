

# Ground thruth vector
g = [0, 1, 0]

# Predicted vector
y = [0.25, 0.6, 0.15]

# Calculate the Hinge loss

# y is the 'y' vector and j is the index of the correct class
def SVM_loss(y, j):
    loss = 0
    for i in range(len(y)):
        if i != j:
            loss += max(0, y[i] - y[j] + 1)
    return loss

# Uses the ground truth vector to get the y index
j = g.index(1)

# Calculate the loss
loss = SVM_loss(y, j)
print(loss)
