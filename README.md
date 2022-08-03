# PAPP
Parallel All Path Pruning for neuron tracing


功能: 对数据进行神经元追踪, 输入为原始图像(tif格式)，输出为swc格式。

数据: data/case1-slide2-section2-left-cell3_merge_c2.tif

结果: results/FastMarching_Resample_GWDT_AfterPruneMerge.swc

主函数：kernel.cu


步骤:
1. 读入图像。
2. 对图像进行阈值化、灰度距离变换(Grey Weight Distance Transform)等预处理。//threshold.cu
3. 对图像进行流压缩，去除图中为0的值。 //compaction.cu
3. 对图像进行泊松盘采样，给神经元追踪提供种子点。//poissonSample.cu
4. 对图像进行初始追踪，从各个种子点开始，使用并行的快速行进(Fast Marching)方法进行扩展。//fastmarching.cu
5. 将各个种子扩展出来的不同分支进行拓扑上的合并。//mergeSegments.cu
6. 将合并后的结果进行剪枝(Pruning)，得到最终追踪结果。//pruing.cu

loadTiff.cpp: 用于读入tif文件

TimerClock.hpp: 用于计时

