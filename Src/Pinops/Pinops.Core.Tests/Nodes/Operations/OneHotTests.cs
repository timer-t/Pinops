using NUnit.Framework;
using Pinops.Core.Nodes;

namespace Pinops.Core.Tests.Nodes.Operations
{
    public class OneHotTests
    {
        [SetUp]
        public void Setup()
        {

        }

        [TestCase(new int[] { 3 }, new float[] { 0, 1, 2 }, ExpectedResult = new float[] { 1, 0, 0,
                                                                                           0, 1, 0,
                                                                                           0, 0, 1 })]
        [TestCase(new int[] { 2, 3 }, new float[] { 0, 1, 2,
                                                    3, 4, 5 }, ExpectedResult = new float[] { 1, 0, 0, 0, 0, 0,
                                                                                              0, 1, 0, 0, 0, 0,
                                                                                              0, 0, 1, 0, 0, 0,
                                                                                              0, 0, 0, 1, 0, 0,
                                                                                              0, 0, 0, 0, 1, 0,
                                                                                              0, 0, 0, 0, 0, 1 })]
        public float[] Forward(int[] indicesTensorShape, float[] indicesArray)
        {
            // Arrange
            var indicesVar = Variable.New(indicesTensorShape);
            indicesVar.Load(indicesArray);

            // Act
            var resultTensor = Pinops.Core.Operations.OneHot(indicesVar).Execute();

            // Assert
            return resultTensor.As1DArray();
        }
    }
}
