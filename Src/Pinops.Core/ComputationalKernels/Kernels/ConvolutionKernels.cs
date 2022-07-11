using ILGPU;
using ILGPU.Algorithms;

namespace Pinops.Core.ComputationalKernels
{
    internal static class ConvolutionKernels
    {
        // src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
        internal static void Im2ColKernel(Index1D index,
                                          ArrayView<float> data_im,
                                          ArrayView<int> parameters,
                                          ArrayView<float> data_col,
                                          int data_im_offset = 0,
                                          int data_col_offset = 0)
        {
            int height = parameters[0],
                width = parameters[1],
                kernel_h = parameters[2],
                kernel_w = parameters[3],
                pad_h = parameters[4],
                pad_w = parameters[5],
                stride_h = parameters[6],
                stride_w = parameters[7],
                dilation_h = parameters[8],
                dilation_w = parameters[9],
                height_col = parameters[10],
                width_col = parameters[11];

            int h_index = index / width_col;
            int h_col = h_index % height_col;
            int w_col = index % width_col;
            int c_im = h_index / height_col;
            int c_col = c_im * kernel_h * kernel_w;
            int h_offset = h_col * stride_h - pad_h;
            int w_offset = w_col * stride_w - pad_w;
            int data_col_ptr = 0;
            data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
            int data_im_ptr = data_im_offset;
            data_im_ptr += (c_im * height + h_offset) * width + w_offset;
            for (int i = 0; i < kernel_h; ++i)
            {
                for (int j = 0; j < kernel_w; ++j)
                {
                    int h_im = h_offset + i * dilation_h;
                    int w_im = w_offset + j * dilation_w;
                    if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                    {
                        data_col[data_col_ptr + data_col_offset] = data_im[data_im_ptr + (i * dilation_h * width + j * dilation_w)];
                    }
                    else
                    {
                        data_col[data_col_ptr + data_col_offset] = 0;
                    }
                    data_col_ptr += height_col * width_col;
                }
            }
        }

        // src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
        internal static void Col2ImKernel(Index1D index,
                                          ArrayView<float> data_col,
                                          ArrayView<int> parameters,
                                          ArrayView<float> data_im,
                                          int data_im_offset = 0)
        {
            int height = parameters[0],
                width = parameters[1],
                kernel_h = parameters[2],
                kernel_w = parameters[3],
                pad_h = parameters[4],
                pad_w = parameters[5],
                stride_h = parameters[6],
                stride_w = parameters[7],
                dilation_h = parameters[8],
                dilation_w = parameters[9],
                height_col = parameters[10],
                width_col = parameters[11];

            float val = 0;
            int w_im = index % width + pad_w;
            int h_im = (index / width) % height + pad_h;
            int c_im = index / (width * height);
            int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
            int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;

            // compute the start and end of the output
            int w_col_start = (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
            int w_col_end = XMath.Min(w_im / stride_w + 1, width_col);
            int h_col_start = (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
            int h_col_end = XMath.Min(h_im / stride_h + 1, height_col);

            for (int h_col = h_col_start; h_col < h_col_end; h_col += 1)
            {
                for (int w_col = w_col_start; w_col < w_col_end; w_col += 1)
                {
                    int h_k = (h_im - h_col * stride_h);
                    int w_k = (w_im - w_col * stride_w);
                    if (h_k % dilation_h == 0 && w_k % dilation_w == 0)
                    {
                        h_k /= dilation_h;
                        w_k /= dilation_w;
                        int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                              height_col + h_col) * width_col + w_col;
                        val += data_col[data_col_index];
                    }
                }
            }
            data_im[index + data_im_offset] = val;
        }
    }
}
