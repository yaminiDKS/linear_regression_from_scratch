# 📈 Linear Regression From Scratch with Streamlit Interface

This repository implements **Linear Regression** using **Gradient Descent** from scratch in Python (no `sklearn`), with full **mathematical formulation**, an **interactive Streamlit UI**, and a focus on understanding core ML concepts visually and numerically.

---

## 🎯 Objective

Given a dataset with features `x` and target `y`, we fit a straight line:

\[
y = mx + b
\]

The goal is to learn the **slope `m`** and **intercept `b`** such that the line best fits the data by minimizing prediction errors.

---

## 🧠 Mathematical Foundations

### 1. **Prediction Function**

\[
\hat{y} = mx + b
\]

---

### 2. **Loss Function: Mean Squared Error (MSE)**

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - (mx_i + b))^2
\]

Where:
- \( y_i \) = true value
- \( \hat{y}_i \) = predicted value
- \( n \) = number of samples

---

### 3. **Gradient Descent Optimization**

We want to minimize MSE by updating parameters in the **direction of steepest descent**:

**Gradient of MSE w.r.t slope `m`:**
\[
\frac{\partial \text{MSE}}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - (mx_i + b))
\]

**Gradient of MSE w.r.t intercept `b`:**
\[
\frac{\partial \text{MSE}}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - (mx_i + b))
\]

---

### 4. **Parameter Updates**

Using a learning rate \( \alpha \), update rules:

\[
m := m - \alpha \cdot \frac{\partial \text{MSE}}{\partial m}
\]
\[
b := b - \alpha \cdot \frac{\partial \text{MSE}}{\partial b}
\]

Repeat for `epochs` number of iterations.

---

## 🧾 Code Architecture

- `LinearRegressionGD` – custom class to:
  - Train using gradient descent
  - Predict new values
  - Track loss over time

- `Streamlit UI` – interactive tool to:
  - Upload datasets
  - Set learning rate and epochs
  - Visualize the regression line and loss curve

---

## 🖼️ Visuals

### Regression Line:
- Scatter plot of data points
- Best-fit line dynamically drawn

### Loss Curve:
- X-axis: Epochs
- Y-axis: MSE Loss

---

## 🧪 Streamlit App: How to Use

### 1. ✅ Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
