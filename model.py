import numpy as np

class LinearRegression():
    def __init__(self, learning_rate = 0.33, iters = 1000):
        self.alpha = learning_rate
        self.iters = iters
        self.weights = None # (w)
        self.bias = None # (b)
    
    
    def compute_cost(self, X, y, w, b):
        """
        Cost function j(w,b)

        Args:
            X (ndarray (m,n): Data, m examples with n features
            y (ndarray (m,)): target values
            w (ndarray (n,)): model parameters
            b (scalar)      : model parameter
        Returns:
            total_cost (scalar) : total cost for w and b parameters.
        """
        total_cost = 0
        n_samples , _ = X.shape
        f_wb = np.dot(X,w) + b
        for m in range(n_samples):
            error = (f_wb[m] - y[m])**2
            total_cost += error
        return total_cost / (2*n_samples)
        

    def compute_gradient(self, X, y, w, b):
        """
        Computes the gradient 
    
        Args:
          X (ndarray (m,n): Data, m examples with n features
          y (ndarray (m,)): target values
          w (ndarray (n,)): model parameters  
          b (scalar)      : model parameter
        Returns:
          dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
          dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
        """
        m,n = X.shape
        dj_dw = np.zeros((n,))                          
        dj_db = 0

        for i in range(m):
            f_wb = np.dot(X[i], w) + b 
            error = f_wb - y[i]
            for j in range(n):
                dj_dw[j] = dj_dw[j] + error * X[i][j]
            dj_db = dj_db + error
                
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw ,dj_db
    
    def gradient_descent(self, X, y, w, b): 
        """
        Performs batch gradient descent

        Args:
          X (ndarray (m,n)   : Data, m examples with n features
          y (ndarray (m,))   : target values
          w (ndarray (n,)): Initial values of model parameters  
          b (scalar)      : Initial values of model parameter

        Returns:
          w (ndarray (n,))   : Updated values of parameters
          b (scalar)         : Updated value of parameter 
        """


        for i in range(self.iters):
            # Calculate the gradient 
            dj_dw ,dj_db = self.compute_gradient(X, y, w, b)   

            # Update parameters using w, b, alpha and gradient
            w = w - self.alpha * dj_dw               
            b = b - self.alpha * dj_db               

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i% np.ceil(self.iters / 10) == 0:
                print(f"Iteration {i:4d}: Cost {self.compute_cost(X, y, w, b)} ")

        return w, b    


    def fit (self, X, y):
        """
        Training of the our model (optimizing the weights(w))

        Args:
            X (ndarray (m,n): Data, m examples with n features
            y (ndarray (m,)): target values
        """
        _ , n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        self.weights, self.bias = self.gradient_descent(X, y, self.weights, self.bias)

    def predict(self, X):
        # y = w.x + b
        return np.dot(X, self.weights) + self.bias

