from __future__ import print_function

import matplotlib
matplotlib.use('TkAgg')
from KalmanBoxTracker import *

"""
当我们设置相同的seed，每次生成的随机数相同
"""
np.random.seed(0)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        """
        这里的tracker就是前面的卡尔曼滤波追踪器的对象列表
        这里有些迷糊，咋没有初始化，而且是一个list的tracker
        这里勉强也能明白，因为每一个tracker都有用
        因为之前消失的目标会再出现，就会用到之前的tracker
        """
        self.trackers = []
        self.frame_count = 0

    def update(self, dets = np.empty((0, 5))):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections
        (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        """
        这里搞了一个的array object在这里
        to_del存放待删除的tracker的id
        ret记录的是最后的tracker的结果
        这里一定要区分，trks中存储的是该帧最后返回的结果
        tracker记录的是以后大概率还会出现的目标的tracker
        trks可以看作是tracker的一个子集
        在跟新过程中会修改tracker，每一帧都会修改
        但是trks是仅仅存在于更新过程中他不是全局变量，就是一个返回结果的暂存器
        可以看出来trks的要求更严格一些
        """
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            """
            这里predict()获取的是二维（n, 4）的array，里n一般是1
            所以pos就是一维的(4,)
            分别是[x1, y1, x2, y2]
            返回的是bbox的形式
            """
            pos = self.trackers[t].predict()[0]
            """
            注意这里是一个trk，不是trks是一个中间变量
            把pos转化成了五列，仍然是array维度是（5，）
            最后增加的一列是否可以看作是置信度
            这里实际上已经对trks继续赋值了
            每一个tracker都得到了一个预测的结果，都要记录下来
            """
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            """
            isnan（）用于测试元素是否为NaN（非数字）,结果返回想同维度的False或者True
            eg：temp = [1 2 3]  np.isnan(temp) = [False  False False]
            np.all(np.array)   对矩阵所有元素做与操作，所有为True则返回True
            np.any(np.array)   对矩阵所有元素做或运算，存在True则返回True
            如果pos都是数字，那么np.any(np.isnan(pos))就是False
            """
            if np.any(np.isnan(pos)):
                to_del.append(t)

        """
        将预测为空的卡尔曼跟踪器所在行删除
        最后trks中存放的是上一帧中被跟踪的所有物体在当前帧中预测的非空bbox
        np.ma.masked_invalid()的作用是把trks中不是数的给遮盖起来
        np.ma.compress_rows()把出现遮盖的行都删掉了
        两句连在一起就达到了删除空bbox的效果
        """
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        """
        这里结合上面进行分析，可以发现to_del记录的是不合法的
        也就是把之前很多的tracker中没有用的剔除掉
        """
        for t in reversed(to_del):
            self.trackers.pop(t)

        """
        经过上面的分析可以看出来
        实际update中检测框是传递进来的
        上面那一大坨是生成tracker预测的结果（预测出本帧的结果）
        然后将预测的bbox和检测生成的bbox建立关联性
        返回匹配的目标矩阵matched, 新增目标的矩阵unmatched_dets, 离开画面的目标矩阵unmatched_trks
        """
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        """
        这里对tracker依然存在的，根据检测的结果对原来的tracker继续更新
        """
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        """
        这里对新生成的目标生成一个tracker，然后更新trackers
        添加进trackers中
        """
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        """
        对新的卡尔曼跟踪器集进行倒序遍历
        """
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            """
            获取trk跟踪器的状态 [x1,y1,x2,y2]
            """
            d = trk.get_state()[0]
            """
            如果一个tracker刚更新过而且最近几帧发生过匹配
            就把他添加到ret中，说明在这一帧最后结果中有他了
            """
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            """
            某一个目标已经好几帧没有出现了
            就把他从tracker中去掉，默认他以后不会再出现
            或者出现后我们把它当作不存在，更不会把他放在本帧的结果中
            """
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if(len(ret)>0):
            return np.concatenate(ret)

        return np.empty((0,5))