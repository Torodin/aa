import matplotlib.pyplot as plt
import numpy as np

data = [
    { "x": 3.5, "y": 18.0},
    { "x": 3.69, "y": 15.0},
    { "x": 3.44, "y": 18.0},
    { "x": 3.43, "y": 16.0},
    { "x": 4.34, "y": 15.0},
    { "x": 4.42, "y": 14.0},
    { "x": 2.37, "y": 24.0},
]

class Predictor:
    def __init__(self, w, b, train_rate):
        self.w = w
        self.b = b
        self.train_rate = train_rate
    
    def predict(self, x):
        return self.b + self.w * x

    def getLoss(self, data):
        lossSum = 0

        for row in data:
            prediction = self.predict(row["x"])
            lossSum += (prediction - row["y"]) ** 2

        return lossSum / len(data)
    
    def getWeightSlope(self, data):
        lossSum = 0

        for row in data:
            prediction = self.predict(row["x"])
            lossSum += (prediction - row["y"]) * (2 * row["x"])

        return lossSum / len(data)

    def getBiasSlope(self, data):
        lossSum = 0

        for row in data:
            prediction = self.predict(row["x"])
            lossSum += (prediction - row["y"]) * 2

        return lossSum / len(data)

    def updateModel(self, data):
        self.w = self.w - (self.getWeightSlope(data) * self.train_rate)
        self.b = self.b - (self.getBiasSlope(data) * self.train_rate)

    def trainModel(self, data, iterations):
        t = np.arange(0., 6., 0.5)
        loss = []

        x_data = [point["x"] for point in data]
        y_data = [point["y"] for point in data]

        fig_reg, axs_reg = plt.subplots(2, 2)
        fig_reg.suptitle("Regression Evolution")
        axs_reg = axs_reg.flatten()

        plotInterval = iterations // 4
        plotCounter = 0

        for i in range(iterations):
            print(f"Iteration {i}")
            print(f"Weight: {self.w} | Bias: {self.b}")

            ecmLoss = self.getLoss(data)
            print(f"ECM Loss: {ecmLoss}")
            loss.append(ecmLoss)

            self.updateModel(data)

            if (i + 1) % plotInterval == 0 and plotCounter < 4:
                ax = axs_reg[plotCounter]
                ax.plot(x_data, y_data, 'ro', label="Examples")
                ax.plot(t, self.predict(t), 'b-', label="Prediction")
                ax.axis([0, 6, 0, 30])
                ax.set_title(f"Iteration {i + 1}")
                ax.legend()

                plotCounter += 1
        
        print(f"Final model weight: {self.w} and bias: {self.b}")
        
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(range(iterations), loss, 'g-')
        ax_loss.set_title("Loss evolution")
        ax_loss.set_xlabel("Iterations")
        ax_loss.set_ylabel("ECM Loss")
        ax_loss.grid(True)

        plt.show()



predictor = Predictor(0, 0, 0.01)

predictor.trainModel(data, 3500)