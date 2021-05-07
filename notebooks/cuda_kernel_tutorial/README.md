In this tutorial, we will cover how to write your own CUDA kernel.

[Nvidia CUDA Tutorial](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

[CUDA Tutorial](http://deeplearnphysics.org/Blog/2018-10-02-Writing-your-own-CUDA-kernel-1.html)

<dl>
  <dt>Kernel</dt>
  <dd>The function to be run by CUDA on the GPU</dd>
  <dt>Thread</dt>
  <dd>Execution code to be run in parallel on the GPU.</dd>
  <dt>Blocks</dt>
  <dd>Threads are grouped into blocks (an abstraction). Typically, a thread block contains up to 1024 threads.
  <dt>Grids</dt>
  <dd>Abstraction that contains thread blocks</dd>
</dl>

