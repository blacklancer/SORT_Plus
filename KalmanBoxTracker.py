from __future__ import print_function
import matplotlib
matplotlib.use('TkAgg')
from filterpy.kalman import KalmanFilter
from utils import *


"""
当我们设置相同的seed，每次生成的随机数相同
"""
np.random.seed(0)


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    """
    记录出想过的对象
    """
    count = 0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model

        """
        状态变量 x 的设定是一个 7维向量：x=[u, v, s, r, u^, v^, s^]T,维度是（7，1）是七行，每行一列
        u、v 分别表示目标框的中心点位置的 x、y 坐标，s 表示目标框的面积，r 表示目标框的宽高比。
        u^、v^、s^ 分别表示横向 u(x方向)、纵向 v(y方向)、面积 s 的运动变化速率。
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        """
        状态转移矩阵 F
        定义的是一个 7x7 的单位方阵，运动形式和转换矩阵的确定都是基于匀速运动模型，状态转移矩阵F根据运动学公式确定，
        跟踪的目标假设为一个匀速运动的目标。通过 7x7 的状态转移矩阵F 乘以 7*1 的状态变量 x 
        即可得到一个更新后的 7x1 的状态更新向量x。
        """
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])

        """
        观测矩阵 H
        定义的是一个 4x7 的矩阵，乘以 7x1 的状态更新向量 x 即可得到一个 4x1 的 [u,v,s,r] 的估计值。
        """
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        """
        测量噪声的协方差矩阵 R：diag([1,1,10,10]T)
        先验估计的协方差矩阵 P：diag([10,10,10,10,1e4,1e4,1e4]T)。1e4：1x10 的 4 次方。
        过程激励噪声的协方差矩阵 Q：diag([1,1,1,1,0.01,0.01,1e-4]T)。
        """
        self.kf.R[2:,2:] *= 10.
        # give high uncertainty to the unobservable initial velocities
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        """
        估算出第一个初值的观测量Z，要把原始的bbox来转化成Z
        """
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        """
        记录距离上次更新的时间间隔
        """
        self.time_since_update = 0

        """
        id和kalmanBoxTracker.count表示一共出现过多少个目标
        因为是跟踪的，需要记录
        """
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.history = []
        """
        总的匹配次数 hit
        连续匹配次数 hit_streak
        """
        self.hits = 0
        self.hit_streak = 0
        """
        这里的age是可以看作是帧数变化
        """
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        """
        每次更新时，总的匹配次数 hit 会加 1，连续匹配次数 hit_streak 也加 1
        而如果一旦出现不匹配的情况时，hit_streak 变量会在 predict 阶段被归 0 而重新计时
        """
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        """
        使用卡尔曼滤波去更新状态
        """
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        """
        原有的面积加上面基的变化率小于0，表示已经消失了
        这样的话面积的变化率就为0
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0

        """
        执行卡尔曼滤波预测操作
        age表示的是第age个时刻，实际上就是视频的帧
        """
        self.kf.predict()
        self.age += 1

        """
        一旦出现不匹配的情况，连续匹配次数被归 0
        """
        if(self.time_since_update>0):
             self.hit_streak = 0

        self.time_since_update += 1
        """
        记录预测的x转化成bbox的结果，记录到history，作为后面的参考
        """
        self.history.append(convert_x_to_bbox(self.kf.x))
        """
        返回的是记录的最后一个bbox，实际上就是
        刚刚预测出来添加进去history的那一个
        其实history也只有两种可能，一种是没有update里面有不止一个array
        一种是update之后之前的history会清空，此时里面只有刚才存入的array
        还要注意的是array是二维的（1，4）
        """
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)