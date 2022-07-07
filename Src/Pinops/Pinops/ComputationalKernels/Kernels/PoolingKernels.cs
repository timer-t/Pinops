using ILGPU;
using ILGPU.Algorithms;

namespace Pinops.Core.ComputationalKernels
{
    // src: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cu
    internal static class PoolingKernels
    {
        internal static void MaxPoolForwardKernel(Index1D index,
                                                  ArrayView<float> bottom_data,
                                                  ArrayView<int> parameters,
                                                  ArrayView<float> top_data,
                                                  ArrayView<float> top_mask)
        {
            int channels = parameters[0],
                height = parameters[1],
                width = parameters[2],
                pooled_height = parameters[3],
                pooled_width = parameters[4],
                kernel_h = parameters[5],
                kernel_w = parameters[6],
                stride_h = parameters[7],
                stride_w = parameters[8],
                pad_h = parameters[9],
                pad_w = parameters[10];

            int pw = index % pooled_width;
            int ph = (index / pooled_width) % pooled_height;
            int c = (index / pooled_width / pooled_height) % channels;
            int n = index / pooled_width / pooled_height / channels;
            int hstart = ph * stride_h - pad_h;
            int wstart = pw * stride_w - pad_w;
            int hend = XMath.Min(hstart + kernel_h, height);
            int wend = XMath.Min(wstart + kernel_w, width);
            hstart = XMath.Max(hstart, 0);
            wstart = XMath.Max(wstart, 0);
            float maxval = -float.MaxValue;
            int maxidx = -1;
            int bottom_slice_ptr = (n * channels + c) * height * width;
            for (int h = hstart; h < hend; ++h)
            {
                for (int w = wstart; w < wend; ++w)
                {
                    if (bottom_data[bottom_slice_ptr + (h * width + w)] > maxval)
                    {
                        maxidx = h * width + w;
                        maxval = bottom_data[bottom_slice_ptr + maxidx];
                    }
                }
            }
            top_data[index] = maxval;
            top_mask[index] = maxidx;
        }

        internal static void MaxPoolBackwardKernel(Index1D index,
                                                   ArrayView<float> top_diff,
                                                   ArrayView<int> parameters,
                                                   ArrayView<float> bottom_diff,
                                                   ArrayView<float> top_mask)
        {
            int channels = parameters[0],
                height = parameters[1],
                width = parameters[2],
                pooled_height = parameters[3],
                pooled_width = parameters[4],
                kernel_h = parameters[5],
                kernel_w = parameters[6],
                stride_h = parameters[7],
                stride_w = parameters[8],
                pad_h = parameters[9],
                pad_w = parameters[10];

            // find out the local index
            // find out the local offset
            int w = index % width;
            int h = (index / width) % height;
            int c = (index / width / height) % channels;
            int n = index / width / height / channels;
            int phstart = (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
            int phend = XMath.Min((h + pad_h) / stride_h + 1, pooled_height);
            int pwstart = (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
            int pwend = XMath.Min((w + pad_w) / stride_w + 1, pooled_width);
            float gradient = 0f;
            int offset = (n * channels + c) * pooled_height * pooled_width;
            int top_diff_slice_ptr = offset;
            int top_mask_slice_ptr = offset;
            for (int ph = phstart; ph < phend; ++ph)
            {
                for (int pw = pwstart; pw < pwend; ++pw)
                {
                    if (top_mask[top_mask_slice_ptr + (ph * pooled_width + pw)] == h * width + w)
                    {
                        gradient += top_diff[top_diff_slice_ptr + (ph * pooled_width + pw)];
                    }
                }
            }
            bottom_diff[index] = gradient;
        }

        internal static void AveragePoolForwardKernel(Index1D index,
                                                      ArrayView<float> bottom_data,
                                                      ArrayView<int> parameters,
                                                      ArrayView<float> top_data)
        {
            int channels = parameters[0],
                height = parameters[1],
                width = parameters[2],
                pooled_height = parameters[3],
                pooled_width = parameters[4],
                kernel_h = parameters[5],
                kernel_w = parameters[6],
                stride_h = parameters[7],
                stride_w = parameters[8],
                pad_h = parameters[9],
                pad_w = parameters[10];

            int pw = index % pooled_width;
            int ph = (index / pooled_width) % pooled_height;
            int c = (index / pooled_width / pooled_height) % channels;
            int n = index / pooled_width / pooled_height / channels;
            int hstart = ph * stride_h - pad_h;
            int wstart = pw * stride_w - pad_w;
            int hend = XMath.Min(hstart + kernel_h, height + pad_h);
            int wend = XMath.Min(wstart + kernel_w, width + pad_w);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = XMath.Max(hstart, 0);
            wstart = XMath.Max(wstart, 0);
            hend = XMath.Min(hend, height);
            wend = XMath.Min(wend, width);
            float aveval = 0f;
            int bottom_slice_ptr = (n * channels + c) * height * width;
            for (int h = hstart; h < hend; ++h)
            {
                for (int w = wstart; w < wend; ++w)
                {
                    aveval += bottom_data[bottom_slice_ptr + (h * width + w)];
                }
            }
            top_data[index] = aveval / pool_size;
        }

        internal static void AveragePoolBackwardKernel(Index1D index,
                                                       ArrayView<float> top_diff,
                                                       ArrayView<int> parameters,
                                                       ArrayView<float> bottom_diff)
        {
            int channels = parameters[0],
                height = parameters[1],
                width = parameters[2],
                pooled_height = parameters[3],
                pooled_width = parameters[4],
                kernel_h = parameters[5],
                kernel_w = parameters[6],
                stride_h = parameters[7],
                stride_w = parameters[8],
                pad_h = parameters[9],
                pad_w = parameters[10];

            // find out the local index
            // find out the local offset
            int w = index % width + pad_w;
            int h = (index / width) % height + pad_h;
            int c = (index / width / height) % channels;
            int n = index / width / height / channels;
            int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
            int phend = XMath.Min(h / stride_h + 1, pooled_height);
            int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
            int pwend = XMath.Min(w / stride_w + 1, pooled_width);
            float gradient = 0f;
            int top_diff_slice_ptr = (n * channels + c) * pooled_height * pooled_width;
            for (int ph = phstart; ph < phend; ++ph)
            {
                for (int pw = pwstart; pw < pwend; ++pw)
                {
                    // figure out the pooling size
                    int hstart = ph * stride_h - pad_h;
                    int wstart = pw * stride_w - pad_w;
                    int hend = XMath.Min(hstart + kernel_h, height + pad_h);
                    int wend = XMath.Min(wstart + kernel_w, width + pad_w);
                    int pool_size = (hend - hstart) * (wend - wstart);
                    gradient += top_diff[top_diff_slice_ptr + (ph * pooled_width + pw)] / pool_size;
                }
            }
            bottom_diff[index] = gradient;
        }
    }
}
