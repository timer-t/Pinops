using NUnit.Framework;
using Pinops.Core.Nodes;

namespace Pinops.Core.Tests.Nodes.Operations
{
    public class ArgmaxTests
    {
        [SetUp]
        public void Setup()
        {

        }

        [TestCase(new int[] { 2, 3, 5 }, // Input tensor shape
                  new float[] { 2, 20, 30, 3, 6,
                                3, 11, 16, 1, 8,
                                14, 45, 23, 5, 27,
                                2, 20, 30, 3, 6,
                                3, 11, 16, 1, 8,
                                14, 45, 23, 5, 27 }, // Input array
                  0, // Axis
                  new int[] { 3, 5 }, // Expected shape
                  ExpectedResult = new float[] { 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0,
                                                 0, 0, 0, 0, 0 })]
        [TestCase(new int[] { 2, 3, 5 }, // Input tensor shape
                  new float[] { 2, 20, 30, 3, 6,
                                3, 11, 16, 1, 8,
                                14, 45, 23, 5, 27,
                                2, 20, 30, 3, 6,
                                3, 11, 16, 1, 8,
                                14, 45, 23, 5, 27 }, // Input array
                  1, // Axis
                  new int[] { 2, 5 }, // Expected shape
                  ExpectedResult = new float[] { 2, 2, 0, 2, 2,
                                                 2, 2, 0, 2, 2 })]
        [TestCase(new int[] { 2, 3, 5 }, // Input tensor shape
                  new float[] { 2, 20, 30, 3, 6,
                                3, 11, 16, 1, 8,
                                14, 45, 23, 5, 27,
                                2, 20, 30, 3, 6,
                                3, 11, 16, 1, 8,
                                14, 45, 23, 5, 27 }, // Input array
                  2, // Axis
                  new int[] { 2, 3 }, // Expected shape
                  ExpectedResult = new float[] { 2, 2, 1,
                                                 2, 2, 1 })]
        [TestCase(new int[] { 2, 3, 5 }, // Input tensor shape
                  new float[] { 2, 20, 30, 3, 6,
                                3, 11, 16, 1, 8,
                                14, 45, 23, 5, 27,
                                2, 20, 30, 3, 6,
                                3, 11, 16, 1, 8,
                                14, 45, 23, 5, 27 }, // Input array
                  -1, // Axis
                  new int[] { 1 }, // Expected shape
                  ExpectedResult = new float[] { 11 })]
        [TestCase(new int[] { 30 }, // Input tensor shape
                  new float[] { 2, 20, 30, 3, 6,
                                3, 11, 16, 1, 8,
                                14, 45, 23, 5, 27,
                                2, 20, 30, 3, 6,
                                3, 11, 16, 1, 8,
                                14, 45, 23, 5, 27 }, // Input array
                  -1, // Axis
                  new int[] { 1 }, // Expected shape
                  ExpectedResult = new float[] { 11 })]
        [TestCase(new int[] { 5 }, // Input tensor shape
                  new float[] { 2, 20, 30, 3, 6 }, // Input array
                  -1, // Axis
                  new int[] { 1 }, // Expected shape
                  ExpectedResult = new float[] { 2 })]
        [TestCase(new int[] { 5 }, // Input tensor shape
                  new float[] { 2, 20, 30, 3, 6 }, // Input array
                  0, // Axis
                  new int[] { 1 }, // Expected shape
                  ExpectedResult = new float[] { 2 })]
        public float[] Forward(int[] inputTensorShape,
                               float[] inputArray,
                               int axis,
                               int[] expectedShape)
        {
            // Arrange
            var inputVar = Variable.New(inputTensorShape);
            inputVar.Load(inputArray);

            // Act
            var resultTensor = Pinops.Core.Operations.Argmax(inputVar, axis).Execute();

            // Assert
            Assert.AreEqual(resultTensor.GetShape(), expectedShape);
            return resultTensor.As1DArray();
        }
    }
}
