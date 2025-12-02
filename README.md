# Net Asset Value (NAV) Prediction Using Gradient Descent

## About
This project predicts the **Net Asset Value (NAV)** of a mutual fund over time using **linear regression** built from scratch with **gradient descent**.  

It also shows how **feature scaling** and **regularization** help the model learn better and avoid overfitting.

---

## Features
- Reads NAV data from an Excel file (`.xlsx`)  
- Preprocesses data:
  - Calculates the days from the starting
  - Handles missing values  
- Applies **Min-Max scaling** on input for better gradient descent  
- Implements **gradient descent** to find optimal weight (`w`) and bias (`b`)  
- Adds **L2 regularization** to prevent overfitting  
- Plots:
  - NAV over time  
  - Cost function over iterations  
- Can predict NAV for any future date


