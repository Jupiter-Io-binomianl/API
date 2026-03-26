import numpy as np
import math
import cv2


def pixel_rgb_to_spherical(r, g, b):####color RGB model change into sphere coordinate
    r_float=r.astype(float)
    g_float=g.astype(float)
    b_float=b.astype(float)

    radius = math.sqrt(r_float** 2 + g_float** 2 + b_float ** 2)
    white_radius=np.sqrt(255**2*3)

    # 方位角 theta
    if r == 0 and g == 0:
        theta = 0  # 当R和G都为0时，角度未定义，设为0
    else:
        theta = math.atan2(b,g)

    # 极角 phi
    if radius == 0:
        phi = 0  # 当半径为0时，极角未定义，设为0
    else:
        phi = math.atan2(r,b)
    r_norm=radius/white_radius
    return r_norm, theta, phi

def matrix_prepare(img_rgb):###repare matrix for do
    R_matrix = img_rgb[:, :, 0]
    G_matrix = img_rgb[:, :, 1]
    B_matrix = img_rgb[:, :, 2]
    return R_matrix, G_matrix, B_matrix


def rgb_matrices_to_spherical_matrices(R_matrix, G_matrix, B_matrix):###do the matrix ierate

    height, width = R_matrix.shape

    # 初始化输出矩阵
    r_matrix = np.zeros((height, width), dtype=np.uint8)
    theta_matrix = np.zeros((height, width), dtype=np.uint8)
    phi_matrix = np.zeros((height, width), dtype=np.uint8)

    # 遍历每个像素
    for i in range(height):
        for j in range(width):
            r_val = R_matrix[i, j]
            g_val = G_matrix[i, j]
            b_val = B_matrix[i, j]

            # 调用转换函数
            r_out, theta_out, phi_out = pixel_rgb_to_spherical(r_val, g_val, b_val)

            # 存储结果
            r_matrix[i, j] = r_out
            theta_matrix[i, j] = theta_out
            phi_matrix[i, j] = phi_out

    return r_matrix, theta_matrix, phi_matrix
def my_SVD(r_matrix, theta_matrix, phi_matrix):####svd seperate
    U_r, S_r, V_r= np.linalg.svd(r_matrix)
    U_t, S_t, V_t= np.linalg.svd(theta_matrix)
    U_p, S_p, V_p= np.linalg.svd(phi_matrix)
    my_result={'R':[U_r,S_r,V_r], 'T':[U_t,S_t,V_t], 'P':[U_p,S_p,V_p]}
    return my_result


# 使用示例
if __name__ == "__main__":

    # 方法1: 从图片读取RGB三个二维矩阵
    image_path = "poop.png"  # 替换为你的图片路径

    try:
        # 读取图片
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"错误: {e}")
        print("请确保图片路径正确！")
    my_candy=matrix_prepare(img_rgb)

    result=rgb_matrices_to_spherical_matrices(my_candy[0], my_candy[1], my_candy[2])
    qq=my_SVD(result[0],result[1],result[2])
    print(qq)


        