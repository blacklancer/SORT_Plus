"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
from configs import *
from sort import *

"""
当我们设置相同的seed，每次生成的随机数相同
"""
np.random.seed(0)

if __name__ == '__main__':
    # all train
    args = parse_args()
    display = args.display
    phase = args.phase

    total_time = 0.0
    total_frames = 0

    # used only for display
    colours = np.random.rand(32, 3)

    """
    带图片的模式
    """
    if (display):
        if not os.path.exists('mot_benchmark'):
            print(
                '\n\tERROR: mot_benchmark link not found!\n\n    '
                'Create a symbolic link to the MOT benchmark\n    '
                '(https://motchallenge.net/data/2D_MOT_2015/#download). '
                'E.g.:\n\n $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    """
    省的预测输出的结果没地方存，先判断再用是个好习惯
    """
    if not os.path.exists('output'):
        os.makedirs('output')

    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    """
    glob.glob()函数将会匹配给定路径下的所有pattern，并以列表形式返回
    这里我测试的时候只留下了一个数据集所以
    glob.glob(pattern) = ['data\\train\\KITTI-17\\det\\det.txt']
    list里面只有一个数据
    """
    for seq_dets_fn in glob.glob(pattern):
        # create instance of the SORT tracker
        mot_tracker = Sort(max_age=args.max_age,
                           min_hits=args.min_hits,
                           iou_threshold=args.iou_threshold)

        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

        with open(os.path.join('output', '%s.txt' % (seq)), 'w') as out_file:
            print("Processing %s." % (seq))

            """
            每一个数据最开始都是帧数也就是int(seq_dets[:, 0].max())
            预测和更新的单位都是帧
            """
            for frame in range(int(seq_dets[:, 0].max())):
                # detection and frame numbers begin at 1
                frame += 1
                total_frames += 1
                """
                下面几行都是或取数据原始的数据处理之后才能使用
                主要就是获取bbox
                主要的操作还是numpy的切片操作
                """
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                dets[:, 2:4] += dets[:, 0:2]


                if (display):
                    fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg' % (frame))
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                start_time = time.time()
                """
                每帧都要根据检测的接过来获取更新之后的trackers
                """
                trackers = mot_tracker.update(dets)

                cycle_time = time.time() - start_time
                total_time += cycle_time

                """
                这里把根据检测更新之后的trackers结果存入到文件中
                """
                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                          file=out_file)
                    if (display):
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                        ec=colours[d[4] % 32, :]))

                if (display):
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
    total_time, total_frames, total_frames / total_time))

    if (display):
        print("Note: to get real runtime results run without the option: --display")