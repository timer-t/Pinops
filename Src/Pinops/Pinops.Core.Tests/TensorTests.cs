using NUnit.Framework;
using System;
using Pinops.Core.Nodes;

namespace Pinops.Core.Tests
{
    public class TensorTests
    {
        [SetUp]
        public void Setup()
        {

        }

        [TestCase(0, new int[] { 3, 3 }, new float[] { 1, 2, 3,
                                                       4, 5, 6,
                                                       7, 8, 9 }, ExpectedResult = new float[] { 1, 2, 3 })]
        [TestCase(1, new int[] { 3, 3 }, new float[] { 1, 2, 3,
                                                       4, 5, 6,
                                                       7, 8, 9 }, ExpectedResult = new float[] { 4, 5, 6 })]
        [TestCase(2, new int[] { 3, 3 }, new float[] { 1, 2, 3,
                                                       4, 5, 6,
                                                       7, 8, 9 }, ExpectedResult = new float[] { 7, 8, 9 })]
        [TestCase(0, new int[] { 3 }, new float[] { 1, 2, 3 }, ExpectedResult = new float[] { 1 })]
        [TestCase(1, new int[] { 3 }, new float[] { 1, 2, 3 }, ExpectedResult = new float[] { 2 })]
        [TestCase(2, new int[] { 3 }, new float[] { 1, 2, 3 }, ExpectedResult = new float[] { 3 })]
        public float[] GetNewTensorThroughIndex_IndexInRange_CheckingNewTensorData(int sourceIndex, int[] sourceShape, float[] sourceData)
        {
            var variable = Variable.New(sourceShape);
            variable.Load(sourceData);
            var tensor = variable.GetTensor();

            return tensor[sourceIndex].As1DArray();
        }

        [TestCase(0, new int[] { 3, 3 }, new float[] { 1, 2, 3,
                                                       4, 5, 6,
                                                       7, 8, 9 }, ExpectedResult = new int[] { 3 })]
        [TestCase(1, new int[] { 3, 3 }, new float[] { 1, 2, 3,
                                                       4, 5, 6,
                                                       7, 8, 9 }, ExpectedResult = new int[] { 3 })]
        [TestCase(2, new int[] { 3, 3 }, new float[] { 1, 2, 3,
                                                       4, 5, 6,
                                                       7, 8, 9 }, ExpectedResult = new int[] { 3 })]
        [TestCase(0, new int[] { 3 }, new float[] { 1, 2, 3 }, ExpectedResult = new int[] { 1 })]
        [TestCase(1, new int[] { 3 }, new float[] { 1, 2, 3 }, ExpectedResult = new int[] { 1 })]
        [TestCase(2, new int[] { 3 }, new float[] { 1, 2, 3 }, ExpectedResult = new int[] { 1 })]
        public int[] GetNewTensorThroughIndex_IndexInRange_CheckingNewTensorShape(int sourceIndex, int[] sourceShape, float[] sourceData)
        {
            var variable = Variable.New(sourceShape);
            variable.Load(sourceData);
            var tensor = variable.GetTensor();

            return tensor[sourceIndex].GetShape();
        }

        [TestCase(-1, new int[] { 3 })]
        [TestCase(3, new int[] { 3 })]
        [TestCase(4, new int[] { 3 })]
        public void GetNewTensorThroughIndex_IndexOutOfRange_ThrowingException(int sourceIndex, int[] sourceShape)
        {
            var variable = Variable.New(sourceShape);
            var tensor = variable.GetTensor();

            Assert.Throws<IndexOutOfRangeException>(() => { var subTensor = tensor[sourceIndex]; });
        }
    }
}
