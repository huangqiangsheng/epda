import numpy as np
import math
def vector_angle(v1,v2):
    return np.math.atan2(np.linalg.det([v1,v2]),np.dot(v1,v2))

# define the angle of the vector p1->p2 with x axis
def line_angle(p1,p2):
    v = p2 - p1
    return  math.atan2(v[1], v[0])

def remove_straight_angles(pts):
    tmppts =[] # remove the same point
    small_num_flag = False
    for iter in range(0, len(pts)-1):
        small_num_flag = False
        v1 = pts[iter+1] - pts[iter]
        if np.sum(v1**2) < 1e-5:
            small_num_flag = True
            continue
        tmppts.append(pts[iter])
    if not small_num_flag:
        tmppts.append(pts[-1])

    newpts = [tmppts[0]]
    if 3 <= len(tmppts):
        for iter in range(1,len(tmppts)-1):
            v1 = tmppts[iter] - tmppts[iter -1]
            v2 = tmppts[iter+1] - tmppts[iter]
            if math.fabs(vector_angle(v1,v2))< 1e-5:
                continue
            else:
                newpts.append(tmppts[iter])
    newpts.append(tmppts[-1])
    return newpts

class Waveguide:
    def __init__(self, pts, width, start_face_angle = None, end_face_angle = None,start_angle = None, end_angle = None):
        self.pts = remove_straight_angles(pts)
        self.width = width
        self.start_point = pts[0]
        self.end_point = pts[-1]
        self.start_angle = line_angle(pts[0],pts[1])  
        self.end_angle = line_angle(pts[-2],pts[-1])         
        if start_face_angle != None:
            self.start_face_angle = start_face_angle
            if start_angle != None:
                self.start_angle = start_angle
            else:
                raise Exception('start_angle without parameter!')
            if self.start_face_angle < self.start_angle:
                self.start_face_angle += math.pi
        else:
            self.start_face_angle = self.start_angle+math.pi/2.0     
        if end_face_angle != None : 
            self.end_face_angle = end_face_angle
            if end_angle != None :
                self.end_angle = end_angle
            else :
                raise Exception('end_angle without parameter!')
            if self.end_face_angle < self.end_angle :
                self.end_face_angle += math.pi
        else:
            self.end_face_angle = self.end_angle+ math.pi/2.0
        self.polygon = []

    def wg_length(self):
        length = 0.0
        for iter in range(1,len(self.pts)):
            length = length + np.norm(self.pts[iter] - self.pts[iter-1])
        return length

    def poly(self):
        pt1s = []
        pt2s = []
        tmp_w = math.fabs(self.width/2.0/math.sin(self.start_face_angle-self.start_angle))
        pt1s.append(np.array([math.cos(self.start_face_angle) * tmp_w + self.pts[0][0], math.sin(self.start_face_angle) * tmp_w + self.pts[0][1]]))
        pt2s.append(np.array([-math.cos(self.start_face_angle) * tmp_w + self.pts[0][0], -math.sin(self.start_face_angle) * tmp_w + self.pts[0][1]]))
        if len(self.pts) >= 2:
            for iter in range(1, len(self.pts)-1):
                pt = self.pts[iter]
                v1 = self.pts[iter] - self.pts[iter - 1]
                v2 = self.pts[iter + 1] - self.pts[iter]
                beta = vector_angle(v1,v2)
                tmp_w = math.fabs(self.width/2.0/math.cos(beta/2))
                line_dir = line_angle(self.pts[iter - 1],pt)
                theta = math.pi/2.0 + beta/2.0 + line_dir
                pt1s.append(np.array([math.cos(theta) * tmp_w + pt[0],math.sin(theta) * tmp_w + pt[1]]))
                pt2s.insert(0,np.array([-math.cos(theta)*tmp_w+pt[0],-math.sin(theta)*tmp_w+pt[1]]))      
        tmp_w = math.fabs(self.width/2.0/math.sin(self.end_face_angle - self.end_angle))
        pt1s.append(np.array([math.cos(self.end_face_angle)*tmp_w+self.pts[-1][0],math.sin(self.end_face_angle)*tmp_w+self.pts[-1][1]]))
        pt2s.insert(0,np.array([-math.cos(self.end_face_angle)*tmp_w+self.pts[-1][0],-math.sin(self.end_face_angle)*tmp_w+self.pts[-1][1]]))
        self.polygon = pt1s + pt2s
        return self.polygon

if __name__ == "__main__":
    import bezier
    import matplotlib.pyplot as plt
    point0 = mp.Vector3(-8.0, 2.0)
    cpoint0 = mp.Vector3(2.0,2.0)
    cpoint1 = mp.Vector3(-2.0,-2.0)
    point1 = mp.Vector3(8.0,-2.0)
    width = 1.0

    factor = np.linspace(0,1,41)

    nodes = np.asfortranarray([[point0.x, cpoint0.x, cpoint1.x, point1.x],[point0.y, cpoint0.y, cpoint1.y, point1.y]])
    curve = bezier.Curve(nodes, degree=3)
    points1 = curve.evaluate_multi(factor)
    new_points = points1.transpose()
    plt.figure(1)

    wg = Waveguide(new_points,1.0)
    poly = wg.poly()
    tmp_poly = np.asarray(poly)
    plt.plot(new_points[:,0],new_points[:,1],'-')
    plt.plot(tmp_poly[:,0],tmp_poly[:,1],'-')
    plt.aix()
    plt.show()