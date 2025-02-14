import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load the Excel file
file_path = "2.xlsx"  # Replace with your file's path
data = pd.read_excel(file_path,skiprows = 4)
data = data[['Net Asset Value', 'NAV date']]
# Display the first few rows of the dataset


# Extract relevant columns
# Assuming the columns are named 'NAV date' and 'Net Asset Value' (adjust as per your file)
data['NAV date'] = pd.to_datetime(data['NAV date'])  # Convert dates to datetime format
data['days_since_start'] = (data['NAV date'] - data['NAV date'].min()).dt.days  # Convert dates to numerical

# Drop any rows with missing values (optional)
data = data.dropna()

mini = 0
maxi = 0
r =0

# Extract the features (X) and target (y)
X = data['days_since_start'].values.reshape(-1, 1)  # Independent variable (days since start)
y = data['Net Asset Value'].values  # Dependent variable (NAV)

def feature_scale(array):
    """
    Scales the features of a 2D NumPy array to a range [0, 1].

    Parameters:
        array (numpy.ndarray): The input 2D array.

    Returns:
        numpy.ndarray: The scaled 2D array.
    """
    # Initialize a scaled array with the same shape as the input
    scaled_array = np.zeros_like(array, dtype=float)

    # Calculate the minimum and maximum values for each column
    min_vals = np.min(array, axis=0)
    max_vals = np.max(array, axis=0)

    # Avoid division by zero
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    r = ranges
    mini = min_vals
    maxi = max_vals
    print(ranges)

    # Apply scaling using nested loops
    rows, cols = array.shape
    for i in range(rows):
        for j in range(cols):
            scaled_array[i, j] = (array[i, j] - min_vals[j]) / ranges[j]

    return scaled_array,ranges


X,r = feature_scale(X)

print(r)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(data['days_since_start'], data['Net Asset Value'], marker='o', linestyle='-', color='b')
plt.title('Net Asset Value over Time')
plt.xlabel('Days Since Start')
plt.ylabel('Net Asset Value')
plt.grid(True)
plt.show()


#cost function

def compute_cost(X,y,w,b,lambda_=1):
  cost = 0.0
  m = X.shape[0]
  for i in range(m):
    f_wb_i = w*X[i]+b
    cost += (f_wb_i - y[i])**2

  # Add the L2 regularization term (lambda_reg * w^2) to the cost
  regularization_term = (lambda_ / 2) * w**2

  # Add regularization to the cost
  cost += regularization_term

  cost/=2*m

  return cost

def compute_grad(X,y,w,b,lambda_ = 1):
  m = X.shape[0]
  dj_dw = 0
  dj_db = 0

  for i in range(m):
    f_wb = w*X[i]+b
    dj_dw_i = (f_wb - y[i]) * X[i]
    dj_db_i = f_wb - y[i]
    dj_dw += dj_dw_i
    dj_db += dj_db_i

  dj_dw += lambda_*w
  dj_dw /= m
  dj_db /= m

  return dj_dw,dj_db

def grad_descent(X,y,w,b,num_iters,alpha):
  cost_history = []

  for i in range(num_iters):
    dj_dw, dj_db = compute_grad(X, y, w , b)
    b = b - alpha * dj_db
    w = w - alpha * dj_dw

    cost = compute_cost(X, y, w, b)
    cost_history.append(cost)

  return w,b, cost_history

def predict(date,w,b):
  date_dt = pd.to_datetime(date)
  min_date = data['NAV date'].min()
  diff = (date_dt - min_date).days
  diff_ans = (diff - mini)  / r
  print(diff,r,diff_ans)

  print("Prediction:", w * diff_ans + b)


#making and running model
# initialize parameters
w_init = 0.01
b_init = 0.01
# some gradient descent settings
iterations = 3000
tmp_alpha = 0.01
# run gradient descent
w_final, b_final,cost_history= grad_descent(X ,y, w_init, b_init, iterations, tmp_alpha)
print(w_final,b_final)


import matplotlib.pyplot as plt

# Plot the cost history
plt.plot(range(iterations), cost_history, label="Cost")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Convergence of Gradient Descent")
plt.legend()
plt.grid()
plt.show()



predict("21st january 2029", w_final,b_final)


