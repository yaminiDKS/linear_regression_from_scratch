 📈 Linear Regression From Scratch with Streamlit Interface
# 📈 Linear Regression from Scratch with Streamlit

This repository implements **Linear Regression** using **Gradient Descent** from scratch in Python (no `sklearn`), with full **mathematical formulation**, an **interactive Streamlit UI**, and a focus on understanding core ML concepts visually and numerically.
This project demonstrates how to build **Linear Regression** from scratch using only **NumPy**, **Pandas**, and **Matplotlib** — without relying on any machine learning libraries like `scikit-learn`. It also features an interactive **Streamlit-based UI**, allowing users to upload their dataset, tweak hyperparameters like **learning rate** and **epochs**, and visually understand how gradient descent affects model learning.

---

## 🎯 Objective

Given a dataset with features `x` and target `y`, we fit a straight line:

\[
y = mx + b
\]

The goal is to learn the **slope `m`** and **intercept `b`** such that the line best fits the data by minimizing prediction errors.
To fit a straight line `y = mx + b` that best approximates the data using **Gradient Descent**, minimizing the prediction error (Mean Squared Error).

---

## 🧠 Mathematical Foundations

### 1. **Prediction Function**

\[
\hat{y} = mx + b
\]

---
### 1. Prediction Function

### 2. **Loss Function: Mean Squared Error (MSE)**
ŷ = mx + b

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - (mx_i + b))^2
\]

Where:
- \( y_i \) = true value
- \( \hat{y}_i \) = predicted value
- \( n \) = number of samples
- `m` = slope (weight)
- `b` = intercept (bias)
- `x` = input feature
- `ŷ` = predicted output

---

### 3. **Gradient Descent Optimization**
### 2. Loss Function: Mean Squared Error (MSE)

We want to minimize MSE by updating parameters in the **direction of steepest descent**:
MSE = (1/n) * Σ (yᵢ - ŷᵢ)²
= (1/n) * Σ (yᵢ - (mxᵢ + b))²

**Gradient of MSE w.r.t slope `m`:**
\[
\frac{\partial \text{MSE}}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - (mx_i + b))
\]

**Gradient of MSE w.r.t intercept `b`:**
\[
\frac{\partial \text{MSE}}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - (mx_i + b))
\]
Where:
- `yᵢ` = true output
- `ŷᵢ` = predicted output
- `n` = number of data points

---

### 4. **Parameter Updates**
### 3. Gradient Descent Optimization

Using a learning rate \( \alpha \), update rules:
To minimize the loss, we compute gradients:

\[
m := m - \alpha \cdot \frac{\partial \text{MSE}}{\partial m}
\]
\[
b := b - \alpha \cdot \frac{\partial \text{MSE}}{\partial b}
\]
**Gradient w.r.t. slope `m`:**

Repeat for `epochs` number of iterations.
∂MSE/∂m = (-2/n) * Σ xᵢ * (yᵢ - (mxᵢ + b))

---

## 🧾 Code Architecture
**Gradient w.r.t. intercept `b`:**

- `LinearRegressionGD` – custom class to:
  - Train using gradient descent
  - Predict new values
  - Track loss over time
∂MSE/∂b = (-2/n) * Σ (yᵢ - (mxᵢ + b))

- `Streamlit UI` – interactive tool to:
  - Upload datasets
  - Set learning rate and epochs
  - Visualize the regression line and loss curve

---

## 🖼️ Visuals
### 4. Parameter Update Rules

Using learning rate `α`, we iteratively update:

m := m - α * ∂MSE/∂m
b := b - α * ∂MSE/∂b

### Regression Line:
- Scatter plot of data points
- Best-fit line dynamically drawn

### Loss Curve:
- X-axis: Epochs
- Y-axis: MSE Loss
Repeat for a fixed number of `epochs` or until the change in loss is below a threshold.

---

## 🧪 Streamlit App: How to Use
## 🧾 Features

### 1. ✅ Setup Environment
- ✅ Linear regression from scratch using NumPy
- ✅ Manual implementation of gradient descent
- ✅ Dynamic control over learning rate and epochs via Streamlit
- ✅ Upload your own datasets with `x` and `y` columns
- ✅ Real-time visualization of regression line and loss curve

```bash
---

## 📤 Sample CSV Format

Make sure your CSV has exactly two columns named `x` and `y`:

```csv
x,y
1,2
2,4
3,5.9
4,8
5,10
📊 Visual Outputs

📍 Regression Line
Scatter plot of the data
Best-fit line using trained parameters
📉 Loss Curve
Tracks MSE across epochs
Shows how loss decreases with learning
🖥 How to Run the App

1. Clone the Repository
git clone https://github.com/yaminiDKS/linear_regression_from_scratch.git
cd linear-regression-from-scratch
2. Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
source venv/bin/activate      # On Windows: venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Launch the Streamlit App
streamlit run linear_regression_streamlit.py
🧪 Project Structure

├── linear_regression_streamlit.py     # Main Streamlit app
├── data.csv                           # Sample dataset
├── requirements.txt                   # Python dependencies
├── README.md                          # This file


📚 Educational Value

This project is ideal for:

Beginners learning ML math
Anyone curious about how gradient descent really works
Students preparing for interviews
Recruiters assessing understanding beyond libraries
🚀 Future Enhancements

 Multivariable linear regression
 Real-time animation of line updates
 Comparison with scikit-learn results
 Downloadable trained parameters