using System;
using System.Collections.Generic;
using System.Text;

namespace Pinops.Core.Structs
{
    public struct MidPointBoundingBox
    {
        public float Probability { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
        public float Width { get; set; }
        public float Height { get; set; }
    }
}
