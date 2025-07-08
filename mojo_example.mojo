
from gpu import thread_idx, block_idx, warp
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from math import iota

# aliases for dtype, blocks and threads per block:
alias dtype = DType.float32
alias threads = 4
alias blocks = 8
alias element_in = blocks * threads

def main():
    var ctx = DeviceContext()

    # initialise input and output buffers
    var in_buffer = ctx.enqueue_create_buffer[dtype](element_in)
    var out_buffer = ctx.enqueue_create_buffer[dtype](blocks)

    # set input and output buffers to right values
    with in_buffer.map_to_host() as bufferio:
        iota(bufferio.unsafe_ptr(), element_in)

    var _ = out_buffer.enqueue_fill(0)

    # layoutTensor creation
    # input
    alias layout = Layout.row_major(blocks, threads)
    # essential to create InTensor type which can be registered by the kernel
    # later
    alias InTensor = LayoutTensor[dtype, layout, MutableAnyOrigin]
    var in_tensor = InTensor(in_buffer)

    alias out_layout = Layout.row_major(blocks)
    # essential to create OutTensor type which can be registered by the kernel
    # later
    alias OutTensor = LayoutTensor[dtype, out_layout, MutableAnyOrigin]
    var out_tensor = OutTensor(out_buffer)
    # kernel with input and output layouttensors as arguments
    # lesson learnt, you need to create the correct tensor type for your kernel
    # with alias as well. That is why the mojo example has the extra alias for
    # In_tensor and Out_tensor.
    fn reduce_sum(in_tensor: InTensor, out_tensor: OutTensor):
        var value = in_tensor.load[1](block_idx.x, thread_idx.x)
        value = warp.sum(value)
        if thread_idx.x == 0:
            out_tensor[block_idx.x] = value


    ctx.enqueue_function[reduce_sum](
        in_tensor,
        out_tensor,
        grid_dim=blocks,
        block_dim=threads,
    )

    with out_buffer.map_to_host() as host_buffer:
        print(host_buffer)

