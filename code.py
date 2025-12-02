import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#loading data
file_path = "2.xlsx"  
data = pd.read_excel(file_path,skiprows = 4)
data = data[['Net Asset Value', 'NAV date']]


#preprocessing
data['NAV date'] = pd.to_datetime(data['NAV date'])  
data['days_since_start'] = (data['NAV date'] - data['NAV date'].min()).dt.days  

# drop any rows with missing values
data = data.dropna()

mini = 0
maxi = 0
r = 0

X = data['days_since_start'].values.reshape(-1, 1)  
y = data['Net Asset Value'].values  


def feature_scale(array):
    # initializing scaled arr
    scaled_array = np.zeros_like(array, dtype=float)

    min_vals = np.min(array, axis=0)
    max_vals = np.max(array, axis=0)

    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1

    # store global values
    global mini, maxi, r
    mini = min_vals
    maxi = max_vals
    r = ranges

    print(ranges)
    
    rows, cols = array.shape
    for i in range(rows):
        for j in range(cols):
            scaled_array[i, j] = (array[i, j] - min_vals[j]) / ranges[j]

    return scaled_array, ranges


# scale X
X, r = feature_scale(X)

print(r)

# plot stuff
plt.figure(figsize=(10, 6))
plt.plot(data['days_since_start'], data['Net Asset Value'], marker='o', linestyle='-', color='b')
plt.title('Net Asset Value over Time')
plt.xlabel('Days Since Start')
plt.ylabel('Net Asset Value')
plt.grid(True)
plt.show()


def compute_cost(X, y, w, b, lambda_=1):
    cost = 0.0
    m = X.shape[0]
    for i in range(m):
        f_wb_i = w * X[i][0] + b   # FIX
        cost += (f_wb_i - y[i])**2
      
    regularization_term = (lambda_ / 2) * (w**2)

    cost += regularization_term

    cost /= (2 * m)

    return cost


def compute_grad(X, y, w, b, lambda_ = 1):
    m = X.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * X[i][0] + b  # FIX
        dj_dw_i = (f_wb - y[i]) * X[i][0]  # FIX
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw += lambda_ * w
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def grad_descent(X, y, w, b, num_iters, alpha):
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_grad(X, y, w, b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

    return w, b, cost_history


def predict(date, w, b):
    date_dt = pd.to_datetime(date)
    min_date = data['NAV date'].min()
    diff = (date_dt - min_date).days

    # FIX: correct scaling using global mini & r
    diff_ans = (diff - mini) / r
    diff_ans = diff_ans[0]   # convert array → scalar

    print(diff, r, diff_ans)

    print("Prediction:", w * diff_ans + b)


# initialize parameters
w_init = 0.01
b_init = 0.01
iterations = 3000
tmp_alpha = 0.01

w_final, b_final, cost_history = grad_descent(X, y, w_init, b_init, iterations, tmp_alpha)
print(w_final, b_final)

# graph of cost history
plt.plot(range(iterations), cost_history, label="Cost")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Convergence of Gradient Descent")
plt.legend()
plt.grid()
plt.show()

#sample case
predict("21st january 2029", w_final, b_final)


