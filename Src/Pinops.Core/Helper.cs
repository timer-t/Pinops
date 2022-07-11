using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Pinops.Core.Structs;

namespace Pinops.Core
{
    public static class Helper
    {
        public static int IndexOfMax(float[] values)
        {
            double max = 0;
            int indexOfMax = 0;
            for (int i = 0; i < values.Length; i++)
                if (values[i] > max)
                {
                    max = values[i];
                    indexOfMax = i;
                }
            return indexOfMax;
        }

        internal static float[] Flatten(float[,] matrix)
        {
            float[] flattenedMatrix = new float[matrix.GetLength(0) * matrix.GetLength(1)];

            for (int h = 0; h < matrix.GetLength(0); h++)
            {
                for (int w = 0; w < matrix.GetLength(1); w++)
                {
                    flattenedMatrix[h * matrix.GetLength(1) + w] = matrix[h, w];
                }
            }

            return flattenedMatrix;
        }

        internal static float[,] ReFlatten(float[] flattenedMatrix, int height, int width)
        {
            float[,] matrix = new float[height, width];

            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    matrix[h, w] = flattenedMatrix[h * width + w];
                }
            }

            return matrix;
        }

        internal static double GetSmallRandomNumber()
        {
            var _random = new Random();
            return (.0009 * _random.NextDouble() + .0001) * (_random.Next(2) == 0 ? -1 : 1);
        }

        internal static float IoU(float box1_mid_x,
                                  float box1_mid_y,
                                  float box1_w,
                                  float box1_h,
                                  float box2_mid_x,
                                  float box2_mid_y,
                                  float box2_w,
                                  float box2_h)
        {
            // To corners format.
            var box1_x1 = box1_mid_x - box1_w / 2;
            var box1_y1 = box1_mid_y - box1_h / 2;
            var box1_x2 = box1_mid_x + box1_w / 2;
            var box1_y2 = box1_mid_y + box1_h / 2;

            var box2_x1 = box2_mid_x - box2_w / 2;
            var box2_y1 = box2_mid_y - box2_h / 2;
            var box2_x2 = box2_mid_x + box2_w / 2;
            var box2_y2 = box2_mid_y + box2_h / 2;

            var x1 = MathF.Max(box1_x1, box2_x1);
            var y1 = MathF.Max(box1_y1, box2_y1);
            var x2 = MathF.Min(box1_x2, box2_x2);
            var y2 = MathF.Min(box1_y2, box2_y2);

            var intersection = Math.Clamp(x2 - x1, 0, float.MaxValue) * Math.Clamp(y2 - y1, 0, float.MaxValue);

            var box1_area = Math.Abs(box1_x2 - box1_x1) * (box1_y2 - box1_y1);
            var box2_area = Math.Abs(box2_x2 - box2_x1) * (box2_y2 - box2_y1);

            var iou = intersection / (box1_area + box2_area - intersection + 1e-6f);

            return iou;
        }

        public static List<MidPointBoundingBox> NonMaxSuppression(MidPointBoundingBox[] bboxes, float iouThreshold, float probabilityThreshold)
        {
            var bboxesList = bboxes.Where(bb => bb.Probability > probabilityThreshold)
                                   .OrderByDescending(bb => bb.Probability)
                                   .ToList();

            var bboxes_after_nms = new List<MidPointBoundingBox>();

            while (bboxesList.Count > 0)
            {
                var chosen_box = bboxesList[0];
                bboxesList.RemoveAt(0);
                bboxesList = bboxesList.Where(b => Helper.IoU(chosen_box.X,
                                                              chosen_box.Y,
                                                              chosen_box.Width,
                                                              chosen_box.Height,
                                                              b.X,
                                                              b.Y,
                                                              b.Width,
                                                              b.Height) < iouThreshold).OrderByDescending(bb => bb.Probability).ToList();
                bboxes_after_nms.Add(chosen_box);
            }

            return bboxes_after_nms;
        }
    }
}
