# InstantTrace
Fast Parallel Neuron Tracing (In CUDA Implementation)

Function: Neuron tracing for optical microscopy(OM) neuron images. 
The input is in .tif format, the output is in .swc format.

Sample data: See the "data" folder. The correspoding ground truth are set in the "ground truth" folder.

Results: The output of the program are stored in the "results" folder.

The main function：kernel.cu

## How to use: see "how to use.txt"

##注意：请使用release_3 版本(在commit里面查询)！最新版可能不稳定！


Steps:
1. Load the image.
2. Preprocessing by thresholding/Grey Weight Distance Transform. //threshold.cu
3. Making stream compaction for image, remove the zero-valued voxels. //compaction.cu
4. Make parallel poisson disk sampling to generate seeds for neuron tracing. //poissonSample.cu
5. Make intial tracing using parallel fast marching algorithm. //fastmarching.cu
6. Make topology merging for the branches extended by different seeds. //mergesegments.cu
7. Make pruning and refinement for the neuron branches, and reach the final neuron tracing result. //pruning.cu

步骤:
1. 读入图像。
2. 对图像进行阈值化、灰度距离变换(Grey Weight Distance Transform)等预处理。//threshold.cu
3. 对图像进行流压缩，去除图中为0的值。 //compaction.cu
3. 对图像进行泊松盘采样，给神经元追踪提供种子点。//poissonSample.cu
4. 对图像进行初始追踪，从各个种子点开始，使用并行的快速行进(Fast Marching)方法进行扩展。//fastmarching.cu
5. 将各个种子扩展出来的不同分支进行拓扑上的合并。//mergeSegments.cu
6. 将合并后的结果进行剪枝(Pruning)，得到最终追踪结果。//pruing.cu

