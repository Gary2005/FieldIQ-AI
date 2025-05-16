import numpy as np
from scipy.optimize import least_squares
import torch

def build_homography(params):
    H = np.array([[params[0], params[1], params[2]],
                  [params[3], params[4], params[5]],
                  [params[6], params[7], params[8]]], dtype=float)
    return H

def project_point(H, point):
    x_img, y_img = point
    vec = np.array([x_img, y_img, 1.0])
    proj = H.dot(vec)
    proj /= proj[2]
    return proj[:2]

def residuals(params, data):
    H = build_homography(params)
    res = []
    for pt in data:
        x_img = pt['x_image']
        y_img = pt['y_image']
        proj = project_point(H, (x_img, y_img))
        
        res.append(proj[0] - pt['x_world'])
        res.append(proj[1] - pt['y_world'])
        

    # norm = 1
    res.append(np.sum(np.square(params)) - 1)
    return np.array(res)

def check_constraints_sufficient(data):
    """检查提供的点是否足够确定单应性"""
    num_constraints = len(data) * 2 + 1
    if num_constraints < 9:
        return False
    return True

def calc_H(data, init = [1, 0, 0, 0, 1, 0, 0, 0, 0]):
    # data = [
    #     {'x_img': 100, 'y_img': 200, 'x_real': 10, 'y_real': 20},
    #     {'x_img': 300, 'y_img': 200, 'x_real': 30, 'y_real': 20},
    #     {'x_img': 100, 'y_img': 400, 'x_real': 10, 'y_real': 40},
    #     {'x_img': 300, 'y_img': 400, 'x_real': 30, 'y_real': 40},
    #     {'x_img': 200, 'y_img': 300, 'x_real': 20},
    #     {'x_img': 250, 'y_img': 350, 'y_real': 35},
    #     {'x_img': 220, 'y_img': 330, 'circle': {'a': 20, 'b': 30, 'R': 15}}
    # ]

    # 检查约束是否足够
    is_sufficient = check_constraints_sufficient(data)
    if not is_sufficient:
        return None, None

    # 初始参数估计
    init_params = np.array(init, dtype=float)
#     init_params = np.array([ 1.79851049e-01,  1.24698406e+02,  3.68272133e+00, -3.75337198e+02,
# 1.22601449e+02,  2.38998544e+02,  8.79825296e-03,  1.18776486e+00], dtype=float)
    
    # 最小二乘优化
    result = least_squares(residuals, init_params, args=(data,))

    # print(data)
    # print("优化结果：", result.x)
    # print("残差平方平均:", np.square(residuals(result.x, data)).mean())  # cost为平方和的一半
    # print("是否成功:", result.success)
    # print("终止原因:", result.message)
    # print(residuals(result.x, data))

    # print("优化得到的参数：", result.x)
    
    H_opt = build_homography(result.x)
    # print("优化后的单应性矩阵：\n", H_opt)
    
    # 测试映射
    # test_img_point = (150, 250)
    # proj_real = project_point(H_opt, test_img_point)
    # print("图像点", test_img_point, "映射到实际平面坐标为：", proj_real)

    return H_opt, np.square(residuals(result.x, data)).mean()

# if __name__ == "__main__":
#     calc_H(
        
#         [{'x_img': 0.0, 'y_img': 0.7855740785598755, 'x_real': 52.5}, {'x_img': 0.49765104055404663, 'y_img': 0.7798425555229187, 'x_real': 52.5}, {'x_img': 1.0, 'y_img': 0.7898889183998108, 'x_real': 52.5}, {'x_img': 0.938671886920929, 'y_img': 0.6345648169517517, 'y_real': -20.16}, {'x_img': 1.0, 'y_img': 0.7915648221969604, 'y_real': -20.16}, {'x_img': 0.9940052032470703, 'y_img': 0.39605554938316345, 'x_real': 36}, {'x_img': 0.0, 'y_img': 0.3944537043571472, 'x_real': 36}, {'x_img': 0.0, 'y_img': 0.6323888897895813, 'x_real': 47}, {'x_img': 0.45617708563804626, 'y_img': 0.6240092515945435, 'x_real': 47}, {'x_img': 0.9406301975250244, 'y_img': 0.6357406973838806, 'x_real': 47}, {'x_img': 0.011187499389052391, 'y_img': 0.37337034940719604, 'circle': {'a': 41.5, 'b': 0.0, 'R': 9.15}}, {'x_img': 0.1171770840883255, 'y_img': 0.3476296365261078, 'circle': {'a': 41.5, 'b': 0.0, 'R': 9.15}}, {'x_img': 0.23302604258060455, 'y_img': 0.3328425884246826, 'circle': {'a': 41.5, 'b': 0.0, 'R': 9.15}}, {'x_img': 0.3309166729450226, 'y_img': 0.3243333399295807, 'circle': {'a': 41.5, 'b': 0.0, 'R': 9.15}}, {'x_img': 0.40709373354911804, 'y_img': 0.32685184478759766, 'circle': {'a': 41.5, 'b': 0.0, 'R': 9.15}}, {'x_img': 0.46378645300865173, 'y_img': 0.33125928044319153, 'circle': {'a': 41.5, 'b': 0.0, 'R': 9.15}}, {'x_img': 0.5491771101951599, 'y_img': 0.34134259819984436, 'circle': {'a': 41.5, 'b': 0.0, 'R': 9.15}}, {'x_img': 0.6144062876701355, 'y_img': 0.35216665267944336, 'circle': {'a': 41.5, 'b': 0.0, 'R': 9.15}}, {'x_img': 0.6782447695732117, 'y_img': 0.36997222900390625, 'circle': {'a': 41.5, 'b': 0.0, 'R': 9.15}}, {'x_img': 0.7409947514533997, 'y_img': 0.3906574249267578, 'circle': {'a': 41.5, 'b': 0.0, 'R': 9.15}}, {'x_img': 0.6108177304267883, 'y_img': 0.0, 'circle': {'a': 0.0, 'b': 0.0, 'R': 9.15}}, {'x_img': 0.5329322814941406, 'y_img': 0.02101851999759674, 'circle': {'a': 0.0, 'b': 0.0, 'R': 9.15}}, {'x_img': 0.4228177070617676, 'y_img': 0.026750000193715096, 'circle': {'a': 0.0, 'b': 0.0, 'R': 9.15}}, {'x_img': 0.3309635519981384, 'y_img': 0.027703704312443733, 'circle': {'a': 0.0, 'b': 0.0, 'R': 9.15}}, {'x_img': 0.21395312249660492, 'y_img': 0.01745370402932167, 'circle': {'a': 0.0, 'b': 0.0, 'R': 9.15}}, {'x_img': 0.12441667169332504, 'y_img': 0.0, 'circle': {'a': 0.0, 'b': 0.0, 'R': 9.15}}]
#         ,
# [ 1.79851049e-01,  1.24698406e+02,  3.68272133e+00, -3.75337198e+02,
# 1.22601449e+02,  2.38998544e+02,  8.79825296e-03,  1.18776486e+00])