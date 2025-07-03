# 📈 Linear Regression from Scratch with Streamlit

This project demonstrates how to build **Linear Regression** from scratch using only **NumPy**, **Pandas**, and **Matplotlib** — without relying on any machine learning libraries like `scikit-learn`. It also features an interactive **Streamlit-based UI**, allowing users to upload their dataset, tweak hyperparameters like **learning rate** and **epochs**, and visually understand how gradient descent affects model learning.

---

## 🎯 Objective

To fit a straight line `y = mx + b` that best approximates the data using **Gradient Descent**, minimizing the prediction error (Mean Squared Error).

---

## 🧠 Mathematical Foundations

### 1. Prediction Function

ŷ = mx + b


Where:
- `m` = slope (weight)
- `b` = intercept (bias)
- `x` = input feature
- `ŷ` = predicted output

---

### 2. Loss Function: Mean Squared Error (MSE)

MSE = (1/n) * Σ (yᵢ - ŷᵢ)²
= (1/n) * Σ (yᵢ - (mxᵢ + b))²


Where:
- `yᵢ` = true output
- `ŷᵢ` = predicted output
- `n` = number of data points

---

### 3. Gradient Descent Optimization

To minimize the loss, we compute gradients:

**Gradient w.r.t. slope `m`:**

∂MSE/∂m = (-2/n) * Σ xᵢ * (yᵢ - (mxᵢ + b))


**Gradient w.r.t. intercept `b`:**

∂MSE/∂b = (-2/n) * Σ (yᵢ - (mxᵢ + b))


---

### 4. Parameter Update Rules

Using learning rate `α`, we iteratively update:

m := m - α * ∂MSE/∂m
b := b - α * ∂MSE/∂b


Repeat for a fixed number of `epochs` or until the change in loss is below a threshold.

---

## 🧾 Features

- ✅ Linear regression from scratch using NumPy
- ✅ Manual implementation of gradient descent
- ✅ Dynamic control over learning rate and epochs via Streamlit
- ✅ Upload your own datasets with `x` and `y` columns
- ✅ Real-time visualization of regression line and loss curve

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
