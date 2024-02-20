import numpy as np

class ProjectionParameter:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def conf_int_edge(self, side='r'):
        if side == 'r':   
            edge_value = self.mean + self.std * 1.96
        elif side == 'l':
            edge_value = self.mean - self.std * 1.96
        else:
            raise Exception(f'Invalid "side" argument: {side}')

        return edge_value
    
class ProjectionModel:
    def __init__(self, f_x, f_y, c_x, c_y,
                 k_1, k_2, k_3, p_1, p_2):
        self.f_x = f_x
        self.f_y = f_y
        self.c_x = c_x
        self.c_y = c_y
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = k_3
        self.p_1 = p_1
        self.p_2 = p_2
    
    def calculate_projection(self, x, y, edge_params=[], side='r'):
        r = np.sqrt(x**2 + y**2)
        
        projection_params = {
            'f_x': self.f_x,
            'f_y': self.f_y,
            'c_x': self.c_x,
            'c_y': self.c_y,
            'k_1': self.k_1,
            'k_2': self.k_2,
            'k_3': self.k_3,
            'p_1': self.p_1,
            'p_2': self.p_2
        }
        
        projection_values = {}
        
        for param_name, param in projection_params.items():
            if param_name in edge_params:
                projection_values[param_name] = param.conf_int_edge(side)
            else:
                projection_values[param_name] = param.mean
        
        f_x_value = projection_values['f_x']
        f_y_value = projection_values['f_y']
        c_x_value = projection_values['c_x']
        c_y_value = projection_values['c_y']
        k_1_value = projection_values['k_1']
        k_2_value = projection_values['k_2']
        k_3_value = projection_values['k_3']
        p_1_value = projection_values['p_1']
        p_2_value = projection_values['p_2']
        
        delta_x = (k_1_value*r**2 + k_2_value*r**4 + k_3_value*r**6)*x + 2*p_1_value*x*y + p_2_value*(r**2 + 2*x**2)
        delta_y = (k_1_value*r**2 + k_2_value*r**4 + k_3_value*r**6)*y + p_1_value*(r**2 + 2*y**2) + 2*p_2_value*x*y
        u = c_x_value + f_x_value*(x + delta_x)
        v = c_y_value + f_y_value*(y + delta_y)
        
        return u, v
        
def main():
    f_x = ProjectionParameter(2359.40946,   0.84200)
    f_y = ProjectionParameter(2359.61091,   0.76171)
    c_x = ProjectionParameter(1370.05852,   1.25225)
    c_y = ProjectionParameter(1059.63818,   0.98041)
    k_1 = ProjectionParameter(-0.06652,     0.00109)
    k_2 = ProjectionParameter(+0.06534,     0.01126)
    k_3 = ProjectionParameter(-0.07555,     0.01126)
    p_1 = ProjectionParameter(+0.00065,     0.00011)
    p_2 = ProjectionParameter(-0.00419,     0.00014)
    
    model = ProjectionModel(f_x, f_y, c_x, c_y, k_1, k_2, k_3, p_1, p_2)
    
    # Task 21
    
    # Image coordinates
    x = 0.64
    y = 0.46
    edge_params = ['f_x']
    
    u_est, v_est = model.calculate_projection(x, y)
    u_true, _ = model.calculate_projection(x, y, edge_params=edge_params)
    
    max_error = u_true - u_est
    print(f'\n### TASK 2.1 RESULTS ###\n\nMaximum Error:\t{max_error}\nTrue:\t\t{u_true}\nEstimated:\t{u_est}\n')

    # Task 22
    edge_params = ['p_1', 'p_2']
    u_true_lhs, v_true_lhs = model.calculate_projection(x, y, edge_params=edge_params, side='l')
    u_true_rhs, v_true_rhs = model.calculate_projection(x, y, edge_params=edge_params, side='r')
    l2_norm_lhs = np.sqrt((u_true_lhs - u_est)**2 + (v_true_lhs - v_est)**2)
    l2_norm_rhs = np.sqrt((u_true_rhs - u_est)**2 + (v_true_rhs - v_est)**2)
    
    print(f'\n### TASK 2.2 RESULTS ###\n\nL2 Norm (lhs):\t{l2_norm_lhs}\nL2 Norm (rhs):\t{l2_norm_rhs}\n')
    
    
if __name__ == "__main__":
    main()
