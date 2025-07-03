import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Linear Regression using Gradient Descent
class LinearRegressionGD:
    def __init__(self, lr=0.0001, epochs=10000, tolerance=1e-6):
        self.lr = lr
        self.epochs = epochs
        self.tolerance = tolerance
        self.m = 0
        self.b = 0
        self.loss_history = []

    def _loss(self, X, y):
        return np.mean((y - (self.m * X + self.b)) ** 2)

    def fit(self, X, y):
        prev_loss = float('inf')
        self.loss_history = []

        for _ in range(self.epochs):
            y_pred = self.m * X + self.b
            error = y - y_pred

            # Gradients
            m_grad = -(2 / len(X)) * np.dot(X, error)
            b_grad = -(2 / len(X)) * np.sum(error)

            # Update
            self.m -= self.lr * m_grad
            self.b -= self.lr * b_grad

            loss = self._loss(X, y)
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tolerance:
                break
            prev_loss = loss

    def predict(self, X):
        return self.m * X + self.b

# Streamlit Interface
st.set_page_config(page_title="Linear Regression Visualizer", layout="centered")
st.title("📈 Linear Regression From Scratch with Visualization")
st.markdown("Upload a CSV file with columns `x` and `y` and adjust hyperparameters.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if 'x' not in data.columns or 'y' not in data.columns:
        st.error("CSV must contain columns named 'x' and 'y'")
    else:
        # Sidebar controls
        st.sidebar.header("Hyperparameters")
        lr = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
        epochs = st.sidebar.slider("Epochs", 100, 10000, 1000, step=100)

        # Prepare data
        X = data['x'].values
        y = data['y'].values

        # Train model
        model = LinearRegressionGD(lr=lr, epochs=epochs)
        model.fit(X, y)
        y_pred = model.predict(X)

        # Metrics
        final_loss = model.loss_history[-1]
        st.success(f"✅ Model trained with Final Loss: **{final_loss:.6f}**")
        st.markdown(f"- Slope (m): **{model.m:.4f}**")
        st.markdown(f"- Intercept (b): **{model.b:.4f}**")

        # Regression Plot
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='black', label='Data Points')
        ax.plot(X, y_pred, color='blue', label='Best Fit Line')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Regression Line")
        ax.legend()
        st.pyplot(fig)

        # Loss Curve
        fig2, ax2 = plt.subplots()
        ax2.plot(model.loss_history, color='red')
        ax2.set_title("Loss over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss (MSE)")
        st.pyplot(fig2)
