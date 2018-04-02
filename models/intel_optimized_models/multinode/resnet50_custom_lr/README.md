## Resnet-50 on ImageNet-1K better-than-SOTA convergence in only 72 epochs on Stampede2


This folder stores the prototxt files that allow training of ResNet-50 on ImageNet-1K using 64 Stampede2 Intel Xeon Platinum 8160 ("Skylake") to better-than-SOTA accuracy in only 72 epochs and 4 hours using multiple learning rate collapses, as described in our paper: **Scale out for large minibatch SGD: Residual network training on ImageNet-1K with improved accuracy and reduced time to train** (https://arxiv.org/pdf/1711.04291.pdf). This is a reduced version compared to the one in the paper, running for **only** 72 epochs in total, hence the slightly lower accuracy.

The training data is expected to be in the local /tmp of each worker. The size of ImageNet LMDB is around 42GB, so it fits on usual local storage.

The training process uses HW more efficiently by running 2 Caffe processes per node, achieving relatively high throughput with a batch size of 32 images per worker. The average throughput is 98.5 img/snode, and 6303 images/s for this 64-node setup.


Training is performed in using several intermediary snapshots, explained in the script below:

	export MLSL_NUM_SERVERS=2
	export MLSL_SERVER_AFFINITY="39,38,43,42"

	# warmup for 5 epochs followed by 20 epochs of LR decay, poly with power 1
	OMP_NUM_THREADS=22 KMP_AFFINITY="granularity=fine,compact,1,0"  mpiexec.hydra -PSM2 -l -n 128 -ppn 2 -f hostfile -genv OMP_NUM_THREADS 22  
	-genv KMP_AFFINITY "granularity=fine,compact,1,0" ./build/tools/caffe train --solver=models/intel_optimized_models/multinode/resnet50_custom_lr/solver_init.prototxt 

	# get resulting model back from worker 0
	result=$(head -n 1 hostfile)
	scp $result:/tmp/resnet_50_64nodes_init_iter_7800.caffemodel .


	# more agressively decrease LR for around 12 epochs, poly with power 2
	OMP_NUM_THREADS=22 KMP_AFFINITY="granularity=fine,compact,1,0"  mpiexec.hydra -PSM2 -l -n 128 -ppn 2 -f hostfile -genv OMP_NUM_THREADS 22  
	-genv KMP_AFFINITY "granularity=fine,compact,1,0" ./build/tools/caffe train --solver=models/intel_optimized_models/multinode/resnet50_custom_lr/solver_cyc1.prototxt 
	--weights=resnet_50_64nodes_init_iter_7800.caffemodel

	# get resulting model back from worker 0
	result=$(head -n 1 hostfile)
	scp $result:/tmp/resnet_50_64nodes_cyc1_iter_3750.caffemodel .

	# re-warmup for 3 wpochs and then decrease LR poly with power 2
	OMP_NUM_THREADS=22 KMP_AFFINITY="granularity=fine,compact,1,0"  mpiexec.hydra -PSM2 -l -n 128 -ppn 2 -f hostfile -genv OMP_NUM_THREADS 22  
	-genv KMP_AFFINITY "granularity=fine,compact,1,0" ./build/tools/caffe train --solver=models/intel_optimized_models/multinode/resnet50_custom_lr/solver_cyc2.prototxt 
	--weights=resnet_50_64nodes_cyc1_iter_3750.caffemodel

        # get resulting model back from worker 0
	result=$(head -n 1 hostfile)
	scp $result:/tmp/resnet_50_64nodes_cyc2_iter_4750.caffemodel .

        # re-warmup for 3 wpochs and then decrease LR poly with power 2
	OMP_NUM_THREADS=22 KMP_AFFINITY="granularity=fine,compact,1,0"  mpiexec.hydra -PSM2 -l -n 128 -ppn 2 -f hostfile -genv OMP_NUM_THREADS 22  
	-genv KMP_AFFINITY "granularity=fine,compact,1,0"  ./build/tools/caffe train --solver=models/intel_optimized_models/multinode/resnet50_custom_lr/solver_cyc3.prototxt 
	--weights=resnet_50_64nodes_cyc2_iter_4750.caffemodel

        # get resulting model back from worker 0
	result=$(head -n 1 hostfile)
	scp $result:/tmp/resnet_50_64nodes_cyc3_iter_4750.caffemodel .
	
	#perform final collapse for 5 epochs (scale/aspect ratio augmentation disabled, weight decay increased, LR decay poly with power 2)
	OMP_NUM_THREADS=22 KMP_AFFINITY="granularity=fine,compact,1,0"  mpiexec.hydra -PSM2 -l -n 128 -ppn 2 -f hostfile -genv OMP_NUM_THREADS 22  
	-genv KMP_AFFINITY "granularity=fine,compact,1,0"  ./build/tools/caffe train --solver=models/intel_optimized_models/multinode/resnet50_custom_lr/solver_collapse.prototxt 
	--weights=resnet_50_64nodes_cyc3_iter_4750.caffemodel ; '


        # get final model back from worker 0
	result=$(head -n 1 hostfile)
	scp $result:/tmp/resnet_50_64nodes_cyc_collapse_iter_1560.caffemodel .



The final model achieves around 76.3%/93.2% top-1/top-5 accuracy. The process takes around 240 minutes.



The resulting model(resnet50_64nodes_cyc_collapse_iter_1560.caffemodel) achieves around 76.3%/93.2% top-1/top-5 accuracy. 

	I0402 13:18:56.347950 30419 caffe.cpp:517] loss = 0.942313 (* 1 = 0.942313 loss)
	I0402 13:18:56.347960 30419 caffe.cpp:517] loss3/top-1 = 0.762781
	I0402 13:18:56.348022 30419 caffe.cpp:517] loss3/top-5 = 0.931862
