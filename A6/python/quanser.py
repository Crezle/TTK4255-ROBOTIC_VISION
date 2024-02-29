import matplotlib.pyplot as plt
import numpy as np
from common import *

class Quanser:
    def __init__(self):
        self.K = np.loadtxt('A6/data/K.txt')
        self.heli_points = np.loadtxt('A6/data/heli_points.txt').T
        self.platform_to_camera = np.loadtxt('A6/data/platform_to_camera.txt')

    def residuals(self, u, weights, yaw, pitch, roll):
        '''
        Args:
        u:          2xM array of detected marker locations
        weights:    1D array of length M, where 1 indicates a valid detection
        yaw:        Yaw angle in radians
        pitch:      Pitch angle in radians
        roll:       Roll angle in radians
        
        Returns:    1D array of length 2M, containing the residuals
        '''
        # Compute the helicopter coordinate frames
        base_to_platform = translate(0.1145/2, 0.1145/2, 0.0)@rotate_z(yaw)
        hinge_to_base    = translate(0.00, 0.00,  0.325)@rotate_y(pitch)
        arm_to_hinge     = translate(0.00, 0.00, -0.050)
        rotors_to_arm    = translate(0.65, 0.00, -0.030)@rotate_x(roll)
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        # Compute the predicted image location of the markers
        p1 = self.arm_to_camera @ self.heli_points[:,:3]
        p2 = self.rotors_to_camera @ self.heli_points[:,3:]
        
        
        # hat_u is a 2xM array of predicted marker locations.
        hat_u = project(self.K, np.hstack([p1, p2]))
        self.hat_u = hat_u # Save for use in draw()

        r = ((hat_u - u)*weights).flatten()
        
        return r

    def residuals_model_A(self, u, weights, p_A):
        '''
        This function returns residuals of model A with respect to the detected marker locations.

        Args:
        u:          2xN array of detected marker locations
        weights:    1D array of length N, where 1 indicates a valid detection
        p_A:        1D array of length 26 + 3xN, containing the parameters

        Returns:    1D array of length 2N, containing the residuals
        '''
        X = p_A[:21].reshape(3, 7)
        X = np.vstack((X, np.ones(7)))
        lengths = p_A[21:26]
        pose = p_A[26:].reshape(-1, 3)
        hat_u = np.zeros_like(u)

        for i in range(len(pose)):
            base_to_platform = translate(lengths[0], lengths[0], 0.0)@rotate_z(pose[i, 0])
            hinge_to_base    = translate(0.00, 0.00,  lengths[1])@rotate_y(pose[i, 1])
            arm_to_hinge     = translate(0.00, 0.00, lengths[2])
            rotors_to_arm    = translate(lengths[3], 0.00, lengths[4])@rotate_x(pose[i, 2])
            base_to_camera   = self.platform_to_camera @ base_to_platform
            hinge_to_camera  = base_to_camera @ hinge_to_base
            arm_to_camera    = hinge_to_camera @ arm_to_hinge
            rotors_to_camera = arm_to_camera @ rotors_to_arm
            
            p1 = arm_to_camera @ X[:,:3]
            p2 = rotors_to_camera @ X[:,3:]

            hat_u[:, i*7:(i+1)*7] = project(self.K, np.hstack([p1, p2]))

        r = ((hat_u - u)*weights).flatten()
            
        return r

    def residuals_model_B(self, u, weights, p_B):
        '''
        This function returns residuals of model B with respect to the detected marker locations.

        Args:
        u:          2xN array of detected marker locations
        weights:    1D array of length N, where 1 indicates a valid detection
        p_B:        1D array of length 39 + 3xN, containing the parameters
        
        Returns:    1D array of length 2M, containing the residuals
        '''
        X = p_B[:21].reshape(3, 7)
        X = np.vstack((X, np.ones(7)))
        angles = p_B[21:30]
        lengths = p_B[30:39]
        pose = p_B[39:].reshape(-1, 3)
        hat_u = np.zeros_like(u)

        for i in range(len(pose)):
            base_to_platform = translate(lengths[0], lengths[1], lengths[2]) @ rotate_x(angles[0]) @ rotate_y(angles[1]) @ rotate_z(angles[2]) @ rotate_z(pose[i, 0])
            arm_to_base      = translate(lengths[3], lengths[4], lengths[5]) @ rotate_x(angles[3]) @ rotate_y(angles[4]) @ rotate_z(angles[5]) @ rotate_y(pose[i, 1])
            rotors_to_arm    = translate(lengths[6], lengths[7], lengths[8]) @ rotate_x(angles[6]) @ rotate_y(angles[7]) @ rotate_z(angles[8]) @ rotate_x(pose[i, 2])
            base_to_camera   = self.platform_to_camera @ base_to_platform
            arm_to_camera    = base_to_camera @ arm_to_base
            rotors_to_camera = arm_to_camera @ rotors_to_arm
            
            p1 = arm_to_camera @ X[:,:3]
            p2 = rotors_to_camera @ X[:,3:]

            hat_u[:, i*7:(i+1)*7] = project(self.K, np.hstack([p1, p2]))

        r = ((hat_u - u)*weights).flatten()
            
        return r

    def residuals_A(self, u, weights, lengths, heli_points, p):
        '''
        This function creates a custom residual function based on model A.

        Args:
        u:          2xM array of detected marker locations
        weights:    1D array of length M, where 1 indicates a valid detection
        lengths:    Lengths for T-matrix
        heli_points:3x7 array of helicopter points
        p:          Euler angles of helicopter in radians
        
        Returns:    1D array of length 2M, containing the residuals
        '''
        heli_points = np.vstack((heli_points.reshape(3, 7), np.ones(7))) 

        base_to_platform = translate(lengths[0], lengths[0], 0.0)@rotate_z(p[0])
        hinge_to_base    = translate(0.00, 0.00,  lengths[1])@rotate_y(p[1])
        arm_to_hinge     = translate(0.00, 0.00, lengths[2])
        rotors_to_arm    = translate(lengths[3], 0.00, lengths[4])@rotate_x(p[2])
        self.base_to_camera   = self.platform_to_camera@base_to_platform
        self.hinge_to_camera  = self.base_to_camera@hinge_to_base
        self.arm_to_camera    = self.hinge_to_camera@arm_to_hinge
        self.rotors_to_camera = self.arm_to_camera@rotors_to_arm

        p1 = self.arm_to_camera @ heli_points[:,:3]
        p2 = self.rotors_to_camera @ heli_points[:,3:]

        hat_u = project(self.K, np.hstack([p1, p2]))

        r = ((hat_u - u)*weights).flatten()
        
        return r

    def residuals_B(self, u, weights, angles, lengths, heli_points, p):
        '''
        This function creates a custom residual function based on model B.

        Args:
        u:          2xM array of detected marker locations
        weights:    1D array of length M, where 1 indicates a valid detection
        angles:     Euler angles in radians for T-matrix
        lengths:    Lengths for T-matrix
        heli_points:3x7 array of helicopter points
        p:          Euler angles of helicopter in radians
        
        Returns:    1D array of length 2M, containing the residuals
        '''
        heli_points = np.vstack((heli_points.reshape(3, 7), np.ones(7))) 

        base_to_platform = translate(lengths[0], lengths[1], lengths[2]) @ rotate_x(angles[0]) @ rotate_y(angles[1]) @ rotate_z(angles[2]) @ rotate_z(p[0])
        arm_to_base      = translate(lengths[3], lengths[4], lengths[5]) @ rotate_x(angles[3]) @ rotate_y(angles[4]) @ rotate_z(angles[5]) @ rotate_y(p[1])
        rotors_to_arm    = translate(lengths[6], lengths[7], lengths[8]) @ rotate_x(angles[6]) @ rotate_y(angles[7]) @ rotate_z(angles[8]) @ rotate_x(p[2])
        base_to_camera   = self.platform_to_camera @ base_to_platform
        arm_to_camera    = base_to_camera @ arm_to_base
        rotors_to_camera = arm_to_camera @ rotors_to_arm

        p1 = arm_to_camera @ heli_points[:,:3]
        p2 = rotors_to_camera @ heli_points[:,3:]

        hat_u = project(self.K, np.hstack([p1, p2]))

        r = ((hat_u - u)*weights).flatten()
        
        return r

    def draw(self, u, weights, image_number):
        I = plt.imread('A6/quanser_image_sequence/data/video%04d.jpg' % image_number)
        plt.imshow(I)
        plt.scatter(*u[:, weights == 1], linewidths=1, edgecolor='black', color='white', s=80, label='Observed')
        plt.scatter(*self.hat_u, color='red', label='Predicted', s=10)
        plt.legend()
        plt.title('Reprojected frames and points on image number %d' % image_number)
        draw_frame(self.K, self.platform_to_camera, scale=0.05)
        draw_frame(self.K, self.base_to_camera, scale=0.05)
        draw_frame(self.K, self.hinge_to_camera, scale=0.05)
        draw_frame(self.K, self.arm_to_camera, scale=0.05)
        draw_frame(self.K, self.rotors_to_camera, scale=0.05)
        plt.xlim([0, I.shape[1]])
        plt.ylim([I.shape[0], 0])
