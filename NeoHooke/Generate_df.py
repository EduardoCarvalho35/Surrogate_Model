import numpy as np
import pandas as pd
import timeit
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split

def random_deformation_gradient(min_stretch, max_stretch, min_shear, max_shear):
    
    suniaxial = np.random.uniform(low=min_stretch, high=max_stretch)
    sshear = np.random.uniform(low=min_shear, high=max_shear)
    sbiaxial1= np.random.uniform(low=min_stretch, high=max_stretch)
    sbiaxial2 = np.random.uniform(low=min_stretch, high=max_stretch) 
    
    # Random stretch factors for uniaxial deformation
    s1 = suniaxial
    s2 = 1/np.sqrt(s1)
    s3 = s2

    # Random stretch factors for simple shear deformation
    r = sshear

    # Random stretch factors for biaxial deformation
    s7 = sbiaxial1
    s8 = sbiaxial2
    s9 = 1.0/(s7*s8)

    # Construct deformation gradients
    F_uniaxial = np.array([[s1, 0, 0],
                           [0, s2, 0],
                           [0, 0, s3]])
    F_shear = np.array([[1, r, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    F_biaxial = np.array([[s7, 0, 0],
                          [0, s8, 0],
                          [0, 0, s9]])

    # Compute random rotation matrices:
    # generates three random rotation matrices using uniformly distributed 
    # random angles with increments of 5 degrees between 0 and 2*pi radians for each axis of rotation. The deformation
    # gradients are then rotated by the random rotation matrices before 
    # being combined to compute the final deformation gradient F

    R1 = random_rotation()
    R2 = random_rotation()
    R3 = random_rotation()
    
    # Rotate deformation gradients
    F_uniaxial = np.matmul(np.matmul(R1, F_uniaxial), R1.T)
    F_shear = np.matmul(np.matmul(R2, F_shear), R2.T)
    F_biaxial = np.matmul(np.matmul(R3, F_biaxial), R3.T)

    # Compute deformation gradient
    F = np.matmul(np.matmul(F_biaxial, F_uniaxial), F_shear)

    return rescale_deformation_gradient(F)

def random_rotation():
    
    # random 0,1, or 2 to define axis of rotation
    axis = np.random.randint(0,3)
    
    # angle of rotation around axis (between 0 and 180 with specified increments)
    angle_increment_degree=5  #angle increment in degree
    angle_increment=angle_increment_degree*np.pi/180
    angle = np.random.choice(np.arange(0,np.pi+angle_increment,angle_increment))   
    
    # Rotation matrix
    # Compute Rotation matrix based on the axis and the angle
    R = np.zeros((3,3))
    if axis == 0: # x-axis
        R = np.array([[1, 0, 0],
                       [0, np.cos(angle), -np.sin(angle)],
                       [0, np.sin(angle), np.cos(angle)]])
    elif axis == 1: # y-axis
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                       [0, 1, 0],
                       [-np.sin(angle), 0, np.cos(angle)]])
    else: # z-axis
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0],
                       [0, 0, 1]])
    return R

def rescale_deformation_gradient(F):
    det = np.linalg.det(F)
    F_rescaled = F / det**(1.0/3.0)
    return F_rescaled

def stress(F,C_10):
    F=np.array(F[0])
    B=np.dot(F,F.T)
    C_10=C_10
    SIGMA_iso=2*C_10*B-2*C_10*(1/3)*np.trace(B)*np.eye(3)
    return (SIGMA_iso)

def generate_df(n,C_10):
    pandarallel.initialize(progress_bar=True)
    df=pd.DataFrame()
    df["F"]=[random_deformation_gradient(0.4, 3.0, -1, 1) for j in range(n)]
    df['SIGMA'] = df.parallel_apply(stress, axis=1,args=(C_10,)) 
    return df

n=50000000  #size of the dataframe
C_10=2  #material parameter for Neo Hooke

start = timeit.default_timer()
df=generate_df(n,C_10)
end = timeit.default_timer()
print('Time (min): ',(end-start)/60)

df.to_pickle(f"GenerateDataframe_F_Geral_{n}.pkl")
file=open(f"GenerateDataframe_F_Geral_{n}_time.txt", "w")
file.write(f"Time (min): {(end-start)/60}")

file.close()
