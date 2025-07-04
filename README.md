# 📈 Linear Regression from Scratch with Streamlit

This project implements **Linear Regression** from the ground up using **NumPy**, **Pandas**, and **Matplotlib**, without any machine learning libraries like `scikit-learn`. It also features a fully interactive **Streamlit interface** that lets users upload their dataset and tweak training parameters like **learning rate** and **epochs**, enabling a hands-on visual learning experience.

---

## 🎯 Objective

To fit a line `y = mx + b` to a dataset by minimizing prediction error using **gradient descent**.

---

## 🧠 Mathematical Foundations

### 1. Prediction Function

ŷ = mx + b


Where:
- `m` = slope  
- `b` = intercept  
- `x` = input feature  
- `ŷ` = predicted output  

---

### 2. Loss Function (Mean Squared Error)

MSE = (1/n) * Σ (yᵢ - ŷᵢ)²
= (1/n) * Σ (yᵢ - (mxᵢ + b))²


Where:
- `yᵢ` = actual output  
- `ŷᵢ` = predicted output  
- `n` = number of samples  

---

### 3. Gradient Descent Optimization

We compute partial derivatives of the loss function with respect to the parameters:

**Gradient w.r.t. slope `m`:**

∂MSE/∂m = (-2/n) * Σ xᵢ * (yᵢ - (mxᵢ + b))


**Gradient w.r.t. intercept `b`:**

∂MSE/∂b = (-2/n) * Σ (yᵢ - (mxᵢ + b))


---

### 4. Parameter Updates

Using a learning rate `α`, we update parameters as:

m := m - α * ∂MSE/∂m
b := b - α * ∂MSE/∂b


Repeat for a predefined number of `epochs` until convergence.

---

## 🚀 Features

- Built using **NumPy** and **Pandas** only
- Custom **gradient descent** training loop
- **Early stopping** based on loss tolerance
- Interactive **Streamlit UI** for live experimentation
- Real-time plots of regression fit and loss curve

---

## 🖼️ Visual Outputs

### ✅ Regression Line
- Scatter plot of input data
- Best-fit line using learned `m` and `b`

### 📉 Loss Curve
- Tracks mean squared error over training epochs

---

## 📁 Project Structure

├── linear_regression_streamlit.py # Streamlit app
├── data.csv # Sample dataset
├── requirements.txt # Dependencies
├── outputs/
│ ├── regression_plot.png # Saved regression plot (optional)
│ └── model_metrics.txt # Learned parameters and loss (optional)
└── README.md


---

## ▶️ How to Run

### 1. Clone and Set Up

```bash
git clone https://github.com/yourusername/linear-regression-from-scratch.git
cd linear-regression-from-scratch

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
2. Launch Streamlit App
streamlit run linear_regression_streamlit.py
📤 Sample CSV Format

The uploaded CSV must have columns named x and y:

x,y
1,2
2,4
3,6
4,8
5,10
🎚️ Streamlit Controls

Learning Rate: Controls how fast the model updates (α)
Epochs: Controls how long to train
Visualizations: Watch how the regression line and loss change with each run
📌 Educational Value

This project is ideal for:

Understanding the math behind regression
Visual learners who benefit from immediate feedback
Beginners exploring machine learning fundamentals
Showcasing algorithmic thinking to recruiters
🛠 Future Enhancements

 Multivariate linear regression
 Real-time animation of line updates
 Loss threshold slider
 Comparison with sklearn model
 Model export and load functionality
