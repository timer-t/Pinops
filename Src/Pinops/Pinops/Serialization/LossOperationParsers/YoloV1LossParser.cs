using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Pinops.Core.Nodes;
using Pinops.Core.Nodes.Operations;
using Pinops.Core.Serialization.Models;

namespace Pinops.Core.Serialization.LossOperationParsers
{
    internal class YoloV1LossParser : LossOperationParser
    {
        protected sealed override string OpType
        {
            get
            {
                return "YoloV1Loss";
            }
        }

        protected sealed override LossOperationProto GetProto(Operation operation)
        {
            var yoloV1Loss = (YoloV1Loss)operation;
            var lossOperationProto = new LossOperationProto()
            {
                Type = yoloV1Loss.GetType().Name,
                ObservedValuesPlaceholderShape = yoloV1Loss.InputNodes[1].GetShape(),
                ObservedValuesPlaceholderOnnxId = yoloV1Loss.InputNodes[1].OnnxId
            };
            lossOperationProto.Attributes.Add("WidthCellsCount", yoloV1Loss.WidthCellsCount.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture));
            lossOperationProto.Attributes.Add("HeightCellsCount", yoloV1Loss.HeightCellsCount.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture));
            lossOperationProto.Attributes.Add("ClassesCount", yoloV1Loss.ClassesCount.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture));
            lossOperationProto.Attributes.Add("BoundingBoxesPerCell", yoloV1Loss.BoundingBoxesPerCell.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture));

            return lossOperationProto;
        }

        protected sealed override Operation ParseAndAddToGraph(Graph graph, LossOperationProto lossOperationProto, List<Node> inputNodes)
        {
            var widthCellsCount = (int)float.Parse(lossOperationProto.Attributes.GetValueOrDefault("WidthCellsCount"), new NumberFormatInfo() { NumberDecimalSeparator = "." });
            var heightCellsCount = (int)float.Parse(lossOperationProto.Attributes.GetValueOrDefault("HeightCellsCount"), new NumberFormatInfo() { NumberDecimalSeparator = "." });
            var classesCount = (int)float.Parse(lossOperationProto.Attributes.GetValueOrDefault("ClassesCount"), new NumberFormatInfo() { NumberDecimalSeparator = "." });
            var boundingBoxesPerCell = (int)float.Parse(lossOperationProto.Attributes.GetValueOrDefault("BoundingBoxesPerCell"), new NumberFormatInfo() { NumberDecimalSeparator = "." });

            return new YoloV1Loss(graph,
                                  inputNodes[0],
                                  (Placeholder)inputNodes[1],
                                  widthCellsCount,
                                  heightCellsCount,
                                  classesCount,
                                  boundingBoxesPerCell);
        }
    }
}
