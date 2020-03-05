import tvm
import numpy as np

# Writing a schedule in TVM
# We will implement a 2D convolution operator typically found in models like
# ResNet-18.

# This file has three TODOs:
# 1. Complete the compute declaration which defines the 2D convolution
# algorithm. You can inspect the slow reference implementation in Python for a
# naive algorithm implementation, or read about the computation structure @
# http://cs231n.github.io/convolutional-networks/#conv

# Note that after this first step, you can verify the correctness of your
# compute declaration without completing the second step by just running this
# script. The default schedule along with the time to run will be printed.

# 2. Schedule the convolution computation after writing the compute declaration
# for better performance. You can look at an example schedule for a similar
# operation (matrix multiply) @
# https://docs.tvm.ai/tutorials/optimize/opt_gemm.html#sphx-glr-tutorials-optimize-opt-gemm-py

# 3. Explain each of the schedule transformations you used and the performance
# improvement of the kernel over the default version.

# These parameters define the shape of the convolution operator (it is important
# that you do not change these as they will impact the consistency of our
# results when we check the performance of your schedule).
# To speed up debugging, you can set the number of input channels to 1
# temporarily to run the reference slow implementation quickly.

# By default we compile all code for Intel x86 CPUs using avx-2
# If you receive an error regarding illegal instructions, try removing the 
# "-mcpu=core-avx2" flag from the target in the call to tvm.build


# Shape definition for the convolution
input_channels = 64
output_channels = 64
kernel_size = 3
input_height = 56
input_width = 56
padding = (1,1)

output_height = (input_height + 2*padding[0] - kernel_size + 1)
output_width = (input_width + 2*padding[1] - kernel_size + 1)

# We define the input in H, W, C (height, width, channels) layout
input_shape = (input_height+2*padding[0], input_width+2*padding[1], input_channels)
# We define the kernel weights in H, W, I, O (kernel height, kernel width, input
# channel, output channel) layout
weight_shape = (kernel_size, kernel_size, input_channels, output_channels)
# Wed define the output in H, W, C (height, width, channels) layout
output_shape = (output_height, output_width, output_channels)

# slow version
# We provide a _very_ slow reference implementation written in Python so that
# you can compare the correctness of your results after writing your TVM compute
# declaration.
def slow(data, weight):
    output = np.empty(output_shape)
    for output_channel in range(0, output_channels):
        print("channel:", output_channel)
        for output_y in range(0, output_height):
            for output_x in range(0, output_width):
                accum = 0.0
                input_y = output_y
                input_x = output_x
                for input_channel in range(0, input_channels):
                    for kernel_y in range(0, kernel_size):
                        for kernel_x in range(0, kernel_size):
                            accum +=\
                            np.float32(data[input_y + kernel_y][input_x + kernel_x][input_channel])\
                            *np.float32(weight[kernel_y][kernel_x][input_channel][output_channel])
                output[output_y][output_x][output_channel] = np.float32(accum)
    return output

def conv2d_nhwc():
    input_placeholder = tvm.te.placeholder(input_shape, name='data')
    weight_placeholder = tvm.te.placeholder(weight_shape, name='weight')
    rc = tvm.te.reduce_axis((0, input_channels), name='rc')
    ry = tvm.te.reduce_axis((0, kernel_size), name='ry')
    rx = tvm.te.reduce_axis((0, kernel_size), name='rx')
    #TODO #1: fill in lambda function to define the compute declaration
    comp = tvm.te.compute((output_height, output_width, output_channels),
        lambda output_y, output_x, output_channel: tvm.te.sum(input_placeholder[output_y + ry][output_x + rx][rc] * weight_placeholder[ry][rx][rc][output_channel], axis = [ry, rx, rc]))#[rc,ry,rx]))
    s = tvm.te.create_schedule(comp.op)
    schedule(s, comp)
    print(tvm.lower(s, [input_placeholder, weight_placeholder, comp], simple_mode=True))
    func = tvm.build(s, [input_placeholder, weight_placeholder, comp], target='llvm -mcpu=core-avx2', name='conv') 
    return func 


def schedule(s, comp):
    yo, xo, co = comp.op.axis
    ry, rx, rc = s[comp].op.reduce_axis
    
    #TODO #2: write the rest of the schedule function
    # The goal is to achieve 2x the performance of the default schedule on your machine.
    bn = 32    
    xo, yo, xi, yi = s[comp].tile(comp.op.axis[0], comp.op.axis[1], bn, bn)
    ko, ki = s[comp].split(xo, factor=4)

    #s[comp].reorder(xo, yo, ko, ki, xi, yi)

    # Vectorization
    s[comp].vectorize(yi)



def main():
    func = conv2d_nhwc()
    data = np.random.random(input_shape).astype('float32')
    weight = np.random.random(weight_shape).astype('float32')
    data_tvm = tvm.nd.array(data)
    weight_tvm = tvm.nd.array(weight)
    output_tvm = tvm.nd.array(np.empty(output_shape).astype('float32'))
    timer = func.time_evaluator(func.entry_name, tvm.cpu(0), min_repeat_ms=100)
    res = timer(data_tvm, weight_tvm, output_tvm)
    # Print statement showing timing information
    
    #TODO #3: report the relative speedup and run time numbers of the schedule kernel and the default schedule kernel
    #explain each of the schedule transformations you used, and how they impacted the performance of the kernel

    # original result : ProfileResult(mean=0.1915324284, results=(0.1915324284,))

    # improved result : ProfileResult(mean=0.09961044999999999, results=(0.09961044999999999,))

    # Vectorization. When the memory access pattern is uniform, the compiler can detect this pattern and pass the continuous memory to vector processor. Using vectorization hint the compiler this pattern, so that we can accelerate it.

    print(res)
    output_tvm_numpy = output_tvm.asnumpy() 
    output = slow(data, weight)
    #print(output_tvm_numpy)
    #print(output)
    np.testing.assert_allclose(output, output_tvm_numpy, rtol=1e-5)

if __name__ == '__main__':
    main()

