using System;
using System.Collections.Generic;
using System.Linq;

namespace Pinops.Core.Nodes.Operations
{
    // src: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/argmax_layer.cpp
    internal class Argmax : Operation
    {
        private readonly Node input;
        private readonly int axis;

        internal Argmax(Graph graph,
                        Node input,
                        int axis) : base(graph, input)
        {
            this.input = input;
            this.axis = axis;

            int[] inputShape = input.GetShape();
            int[] outputShape;
            if (axis >= 0 && inputShape.Length > 1)
            {
                outputShape = inputShape.Where((source, index) => index != axis).ToArray();
            }
            else
            {
                outputShape = new int[] { 1 };
            }

            Output = new Tensor(this, outputShape);
        }

        internal override void Forward()
        {
            var inputData = input.Output.As1DArray();
            var outputData = new float[Output.Length];

            RunArgmax(bottom_shape: input.GetShape(),
                      bottom: inputData,
                      top: outputData,
                      axis_: axis);

            Output.Load(outputData);
        }

        internal override void Backward()
        {
            throw new NotImplementedException();
        }

        private void RunArgmax(int[] bottom_shape,
                               float[] bottom,
                               float[] top,
                               int axis_ = -1,
                               int top_k_ = 1,
                               bool out_max_val_ = false)
        {
            int dim, axis_dist;
            bool has_axis_ = axis_ > -1;
            if (has_axis_)
            {
                dim = bottom_shape[axis_];
                // Distance between values of axis in blob
                axis_dist = bottom_shape.Skip(axis_).Aggregate((a, b) => a * b) / dim;
            }
            else
            {
                dim = bottom_shape.Aggregate((a, b) => a * b);
                axis_dist = 1;
            }
            int num = bottom_shape.Aggregate((a, b) => a * b) / dim;
            List<KeyValuePair<float, int>> bottom_data_vector = new List<KeyValuePair<float, int>>(new KeyValuePair<float, int>[dim]);
            for (int i = 0; i < num; ++i)
            {
                for (int j = 0; j < dim; ++j)
                {
                    var bottom_index = (i / axis_dist * dim + j) * axis_dist + i % axis_dist;
                    bottom_data_vector[j] =
                        new KeyValuePair<float, int>(
                            bottom[bottom_index], j);
                }
                bottom_data_vector =
                    bottom_data_vector.OrderByDescending(kvp => kvp.Key)
                                      .ThenBy(kvp => kvp.Value)
                                      .ToList();
                for (int j = 0; j < top_k_; ++j)
                {
                    if (out_max_val_)
                    {
                        if (has_axis_)
                        {
                            // Produces max_val per axis
                            top[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist]
                              = bottom_data_vector[j].Key;
                        }
                        else
                        {
                            // Produces max_ind and max_val
                            top[2 * i * top_k_ + j] = bottom_data_vector[j].Value;
                            top[2 * i * top_k_ + top_k_ + j] = bottom_data_vector[j].Key;
                        }
                    }
                    else
                    {
                        // Produces max_ind per axis
                        var top_index = (i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist;
                        top[top_index]
                          = bottom_data_vector[j].Value;
                    }
                }
            }
        }
    }
}
