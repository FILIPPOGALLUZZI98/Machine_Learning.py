# Algorithm for computing the regularized cost (using lambda) for linear regression
def cost_regularized_linear(X, y, w, b, lambda_ = 1):
    m  = X.shape[0]
    n  = len(w)
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b                                   
        cost = cost + (f_wb_i - y[i])**2                                            
    cost = cost / (2 * m)                                                
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          
    reg_cost = (lambda_/(2*m)) * reg_cost                              
    total_cost = cost + reg_cost                                       
    return total_cost  


# Algorithm for computing the regularized cost (using lambda) for logistic regression
def cost_regularized_logistic(X, y, w, b, lambda_ = 1):
    m,n  = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b                                      
        f_wb_i = sigmoid(z_i)                                         
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)     
    cost = cost/m                                                     
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                         
    reg_cost = (lambda_/(2*m)) * reg_cost                                 
    total_cost = cost + reg_cost                                       
    return total_cost    


# Algorithm for computing the regularized gradient for linear regression
def gradient_regularized_linear(X, y, w, b, lambda_): 
    m,n = X.shape           
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]                 
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]               
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m   
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]
    return dj_db, dj_dw


# Algorithm for computing the regularized gradient for logistic regression
def gradient_regularized_logistic(X, y, w, b, lambda_): 
    m,n = X.shape
    dj_dw = np.zeros((n,))                            
    dj_db = 0.0                                       
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          
        err_i  = f_wb_i  - y[i]                       
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   
    dj_db = dj_db/m                                   
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]
    return dj_db, dj_dw 



