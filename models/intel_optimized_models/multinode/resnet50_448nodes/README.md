## Resnet-50 on ImageNet-1K SOTA convergence in 1 hour using 448 Skylake nodes from Stampede2

This folder stores the prototxt files that allow training of ResNet-50 on ImageNet-1K at > 76% top-1 accuracy using 448 Stampede2 Intel Xeon Platinum 8160 ("Skylake").
The training is performed in about 60 minutes.

The training uses HW more efficiently by running 2 Caffe processes per node, achieving relatively high throughput with a batch size of 16 images per worker. This results in a batch size of 14336

Training is performed in 2 parts:

Part 1.
- warm-up for 5 epochs (446 iterations)
- linear learning rate decay for the following 85 epochs (another 7594 iterations)

This is achieved by:

	OMP_NUM_THREADS=22 KMP_AFFINITY="granularity=fine,compact,1,0"  mpiexec.hydra -PSM2 -l -n 896 -ppn 2 -f hostfile -genv OMP_NUM_THREADS 22  -genv KMP_AFFINITY "granularity=fine,compact,1,0" ./build/tools/caffe train --solver=models/intel_optimized_models/multinode/resnet_50_448nodes/solver.prototxt 


The resulting model achieves around 75.6%/92.7% top-1/top-5 accuracy. The process takes around 60 minutes.


Part 2.
- the model saved at epoch 85 is trained for 5 more epochs, in a collapsed fashion: scale/aspect ratio augmentation disabled, weight decay doubled, learning rate decay with a power of 2.

This is achieved by:

	OMP_NUM_THREADS=22 KMP_AFFINITY="granularity=fine,compact,1,0"  mpiexec.hydra -PSM2 -l -n 896 -ppn 2 -f hostfile -genv OMP_NUM_THREADS 22  -genv KMP_AFFINITY "granularity=fine,compact,1,0"  ./build/tools/caffe train --solver=models/intel_optimized_models/multinode/resnet50_448nodes/solver_collapse.prototxt --weights=resnet_50_448nodes_iter_7638.prototxt

The resulting model(resnet50_448nodes_coll_iter_402.caffemodel) achieves around 76.1%/93.2% top-1/top-5 accuracy. The process takes around 5 minutes.



