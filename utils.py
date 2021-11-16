from __future__ import print_function

import numpy as np
import matplotlib
matplotlib.use('TkAgg')

"""
当我们设置相同的seed，每次生成的随机数相同
"""
np.random.seed(0)



def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)

        """
        使用lap.lapjv实现线性分配（用来作为匈牙利算法的实现）
        可以看作是多任务分配的最优结果，使之代价最小
        _: 赋值的代价，如果return_cost为False，则不返回
        x: 一个大小为n的数组，用于指定每一行被分配到哪一列
        y: 一个大小为n的数组，用于指定每列被分配到哪一行
        其实单独获取x或者y也能获取全局最优的结果
        """
        return np.array([[y[i], i] for i in x if i >= 0])

    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)

        """
        这里是使用匈牙利算法计算最小的代价矩阵
        x 开销矩阵对应的行索引
        y 对应行索引的最优指派的列索引
        zip(x, y)是组成对应的索引比如(x[0], y[0])
        list(zip(x, y))把结果转化成list形式
        np.array()把原来的结果转化成array object
        shape是（n, 2）n是原来list的长度
        """
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    """
    np.expand_dims(bb_gt, axis)
    意思就是增加了一个维度，新增加的维度是第axis，后面的维度都后移一步
    一般来说，增加的这个维度的数值应该就是1
    """

    # print(bb_test.shape)
    # print(bb_gt.shape)

    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)


    """
    np.maximum(X, Y) 用于逐元素比较两个array的大小,取较大的值
    np.minimum(X, Y) 用于逐元素比较两个array的大小,取较小的值
    前面添加维度对这里没有影响，因为[x1,y1,x2,y2]是最后一个维度的数据
    np的切片操作还是很好的
    这里切片操作之后维度，这里只对最后一个维度切片，取得是最后一个
    维度的第一个值，相当于把最后一个维度去掉了
    （2, 1, 5） => (2, 1)，这个一定要注意了
    哪怕切片之后就两个值，他也不是一维的，而是二维的
    """
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    """
    np.maximum(x, Y)用于逐个比较array Y中元素和x大小,取较大的值
    """
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)


    """
    这里是两个维度相等的array object相乘
    结果是[w[.., 0]*h[..,0], w[..,1]*h[..,1],...,w[..,n]*h[..,n]]
    """
    wh = w * h

    """
    这改写一下，使用变量存储两个bb的并集面积
    这里都是相同维度的array object操作，可以看作是对
    单个的数进行操作，然后放在array object的对应位置
    """
    b_wh = ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)

    o = wh / b_wh

    # print(o.shape)
    # print("""""")

    """
    这里的括号可加可不加
    """
    return(o)


"""
把bbox转化为卡尔曼滤波中的观测值z
最后一部是把维度从（4）转化成（4，1）
这里实际上是从一维转化成了二维
"""
def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    # scale is just area
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


"""
把预测值[x,y,s,r]转化成[x1,y1,x2,y2]的形式
根据score的形式返回不同的形式,最后返回的是二维的形式（1，4）或者（1，5）
"""
def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    """
    它是将 tracker 输出的预测框（注意是先验估计值）和 detector 输出的检测框相关联匹配起来。
    输入是 dets： [[x1,y1,x2,y2,score],…] 和 trks： [[x1,y1,x2,y2,tracking_id],…] 
    以及一个设定的 iou 阈值，该门槛是为了过滤掉那些低重合度的目标。
    """

    """
    该过程返回：matches（已经匹配成功的追踪器）
    unmatched_detections（没有匹配成功的检测目标） 
    unmatched_trackers（没有匹配成功的跟踪器）
    """

    # print(detections)
    # print(trackers)
    # print("""""""")

    if(len(trackers)==0):
        """
        np.empty((0, 2), dtype=int)
        零行，每行2个元素”，其中“每行2个元素”没有太大的意义，因为没有行
        这里如果trackers没有内容，仍然需要返回三个东西
        但是这里matches（已经匹配成功的追踪器）unmatched_trackers（没有匹配成功的跟踪器）都为空
        unmatched_detections（没有匹配成功的检测目标）是要返回所有的id的
        这里不太明白为什么unmatched_tracker的形式是（0，5）
        因为后面当unmatched_trackers不为空时，返回的也不是（n，5）格式的
        """
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    """
    detection 是 detector 输出的检测框
    tracker 是 tracker 输出的预测框（注意是先验估计值）
    先update得到tracker,然后检测得到detection
    然后匹配
    此时detection和tracker维度，这里是要是行数不对应
    此时输出的iou矩阵的维度是（d， t）
    d是detection的第一个维度
    t是tracker的第一个维度
    """
    iou_matrix = iou_batch(detections, trackers)

    """
    这里iou_matrix的维度可以是（d， t）
    但是存在维度的数值为0的情况，这时候就没什么意义
    """
    if min(iou_matrix.shape) > 0:
        """
        iou矩阵有效时
        1.先提出iou矩阵中小于阈值的iou值，然后转化成np.int32
        这里转化之后的a矩阵维度和iou矩阵一致，但是所有非0元素都转化成了1
        这里可以看出来，sort使用的是不是匈牙利算法，而是KM算法
        https://zhuanlan.zhihu.com/p/62981901两种区别可从这里看出来
        """
        a = (iou_matrix > iou_threshold).astype(np.int32)
        """
        根据a矩阵的情况进行判断
        a的两个维度中每一行列先求和，然后取最大值
        sum（axis = 0）是计算列的和，sum（axis = 1）计算的是行的和
        1、如果最大值均为1，说明每行每列都只有一个1，直接满足多任务的分配情况，
        这时候是不冲突的，直接就不用使用KM算法进行计算
        2.如果多任务分配需要进一步的计算，就使用之前的iou矩阵去计算
        这里使用KM是从这里体现出来的，如果不使用权重仅仅使用0-1矩阵就是
        使用的匈牙利算法
        两种结果返回的都是二维的array object坐标表示
        """

        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            """
            这一行代码太顶了
            第一步是np.where(a)，获取值不为0的坐标，使用两个array对应存储
            第二步是使用stack来对两个array进行组合最终生成的是和使用KM算法
            得到的结果一致
            """
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            """
            这里传进去的负的iou矩阵，目的就是在多任务分配‘
            尽量保存比较大的iou
            """
            matched_indices = linear_assignment(-iou_matrix)
    else:
        """
        当iou矩阵中有维度的数值是0时，就是直接返回空的match_indices
        """
        matched_indices = np.empty(shape=(0, 2))

    """
    前期生成的matched_indices中，有两个维度
    其切片之后[:, 0]表示匹配上的detection的标记
    如果detection中的d（类似于id）不在最后的matched_indices中，就说明
    这里检测生成的detection没有匹配上
    """
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)

    """
    其切片之后[:, 1]表示匹配上的tracker的标记
    如果tracker中的t（类似于id）不在最后的matched_indices中，就说明
    这里基于预测生成的tracker没有匹配上
    """
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            """
            上面是使用阈值过滤掉iou比较低的匹配
            同时将筛选掉的给加入到未匹配的detection和tracker中去
            """
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            """
            如果过滤不掉，就说明可以匹配的上，这里就加入到匹配结果中
            注意match的格式是二维的（n， 2）表示匹配的格式
            [:, 0]表示匹配的detection的id，[:, 1]表示匹配的tracker的id
            这里append之前需要reshape成维度为（1， 2）的array object
            这时的matchs是一个存储array object的list
            """
            matches.append(m.reshape(1, 2))

    if(len(matches)==0):
        matches = np.empty((0, 2), dtype=int)
    else:
        """
        np.concatenate(matches, axis=0)
        执行的操作就是在第0维上对matchs中的array object进行拼接
        """
        matches = np.concatenate(matches, axis=0)
    """
    注意要把unmatched_detections和unmatched_trackers从list转化成numpy格式
    """
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)