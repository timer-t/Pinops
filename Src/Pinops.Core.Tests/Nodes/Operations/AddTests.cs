using NUnit.Framework;
using System;
using System.Linq;

namespace Pinops.Core.Tests.Nodes.Operations
{
    public class AddTests
    {
        [SetUp]
        public void Setup()
        {

        }

        [Test]
        public void Forward__SameRank2_DifferentShape_TensorsAreCompatible__True()
        {
            // Arrange
            var graph = new Graph();

            var var1 = graph.Variable(1, 3);
            var1.Load(new float[] { 1, 2, 3 });

            var var2 = graph.Variable(3, 1);
            var2.Load(new float[] { 4, 5, 6 });

            var addOp = graph.Add(var1, var2);

            // Act
            var result = graph.Run(addOp).As1DArray();

            // Assert
            Assert.IsTrue(result.SequenceEqual(new float[] { 5, 6, 7,
                                                             6, 7, 8,
                                                             7, 8, 9 }));
        }

        [Test]
        public void Forward__SameRank4_DifferentShape_TensorsAreCompatible__True()
        {
            // Arrange
            var graph = new Graph();

            var var1 = graph.Variable(1, 1, 2, 2);
            var1.Load(new float[] { 0, 0, 0, 0 });

            var var2 = graph.Variable(2, 2, 1, 1);
            var2.Load(new float[] { 1, 2, 3, 4 });

            var var3 = graph.Variable(1, 1, 2, 3);
            var3.Load(new float[] { 0, 0, 0, 0, 0, 0 });

            var var4 = graph.Variable(2, 2, 1, 1);
            var4.Load(new float[] { 1, 2, 3, 4 });

            var addOp1 = graph.Add(var1, var2);
            var addOp2 = graph.Add(var3, var4);

            // Act
            var result1 = graph.Run(addOp1).As1DArray();
            var result2 = graph.Run(addOp2).As1DArray();

            // Assert
            Assert.IsTrue(result1.SequenceEqual(new float[] { 1, 1, 1, 1,
                                                              2, 2, 2, 2,
                                                              3, 3, 3, 3,
                                                              4, 4, 4, 4 }));
            Assert.IsTrue(result2.SequenceEqual(new float[] { 1, 1, 1, 1, 1, 1,
                                                              2, 2, 2, 2, 2, 2,
                                                              3, 3, 3, 3, 3, 3,
                                                              4, 4, 4, 4, 4, 4 }));
        }

        [Test]
        public void Forward__SameRank4_SameShape_TensorsAreCompatible__True()
        {
            // Arrange
            var graph = new Graph();

            var var1 = graph.Variable(2, 2, 2, 2);
            var1.Load(new float[] { 1, 1, 1, 1,
                                    2, 2, 2, 2,
                                    3, 3, 3, 3,
                                    4, 4, 4, 4 });

            var var2 = graph.Variable(2, 2, 2, 2);
            var2.Load(new float[] { 1, 1, 1, 1,
                                    2, 2, 2, 2,
                                    3, 3, 3, 3,
                                    4, 4, 4, 4 });

            var addOp = graph.Add(var1, var2);

            // Act
            var result = graph.Run(addOp).As1DArray();

            // Assert
            Assert.IsTrue(result.SequenceEqual(new float[] { 2, 2, 2, 2,
                                                             4, 4, 4, 4,
                                                             6, 6, 6, 6,
                                                             8, 8, 8, 8 }));
        }

        [Test]
        public void Forward__DifferentRanks_DifferentShape_TensorsAreCompatible__True()
        {
            // Arrange
            var graph = new Graph();

            var var1 = graph.Variable(2, 2);
            var1.Load(new float[] { 0, 0, 0, 0 });

            var var2 = graph.Variable(2, 2, 1, 1);
            var2.Load(new float[] { 1, 2, 3, 4 });

            var var3 = graph.Variable(2, 2, 1, 1);
            var3.Load(new float[] { 1, 2, 3, 4 });

            var var4 = graph.Variable(2, 2);
            var4.Load(new float[] { 0, 0, 0, 0 });

            var var5 = graph.Variable(1, 1, 2, 2, 3);
            var5.Load(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

            var var6 = graph.Variable(2, 2, 1, 1);
            var6.Load(new float[] { 1, 2, 3, 4 });

            var var7 = graph.Variable(2, 1, 2, 2, 3);
            var7.Load(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

            var var8 = graph.Variable(2, 2, 1, 1);
            var8.Load(new float[] { 1, 2, 3, 4 });

            var addOp1 = graph.Add(var1, var2);
            var addOp2 = graph.Add(var3, var4);
            var addOp3 = graph.Add(var5, var6);
            var addOp4 = graph.Add(var7, var8);

            // Act
            var tensor1 = graph.Run(addOp1);
            var tensor2 = graph.Run(addOp2);
            var tensor3 = graph.Run(addOp3);
            var tensor4 = graph.Run(addOp4);
            var result1 = tensor1.As1DArray();
            var result2 = tensor2.As1DArray();
            var result3 = tensor3.As1DArray();
            var result4 = tensor4.As1DArray();

            // Assert
            Assert.IsTrue(result1.SequenceEqual(new float[] { 1, 1, 1, 1,
                                                              2, 2, 2, 2,
                                                              3, 3, 3, 3,
                                                              4, 4, 4, 4 }));
            Assert.IsTrue(result2.SequenceEqual(new float[] { 1, 1, 1, 1,
                                                              2, 2, 2, 2,
                                                              3, 3, 3, 3,
                                                              4, 4, 4, 4 }));
            Assert.IsTrue(result3.SequenceEqual(new float[] { 2, 3, 4, 5, 6, 7,
                                                              9, 10, 11, 12, 13, 14,
                                                              4, 5, 6, 7, 8, 9,
                                                              11, 12, 13, 14, 15, 16 }));
            Assert.IsTrue(result4.SequenceEqual(new float[] { 2, 3, 4, 5, 6, 7,
                                                              9, 10, 11, 12, 13, 14,
                                                              4, 5, 6, 7, 8, 9,
                                                              11, 12, 13, 14, 15, 16,
                                                              2, 3, 4, 5, 6, 7,
                                                              9, 10, 11, 12, 13, 14,
                                                              4, 5, 6, 7, 8, 9,
                                                              11, 12, 13, 14, 15, 16 }));

            Assert.IsTrue(tensor1.GetShape().SequenceEqual(new int[] { 2, 2, 2, 2 }));
            Assert.IsTrue(tensor2.GetShape().SequenceEqual(new int[] { 2, 2, 2, 2 }));
            Assert.IsTrue(tensor3.GetShape().SequenceEqual(new int[] { 1, 2, 2, 2, 3 }));
            Assert.IsTrue(tensor4.GetShape().SequenceEqual(new int[] { 2, 2, 2, 2, 3 }));
        }

        [Test]
        public void Forward__SameRank4_DifferentShape_TensorsAreNotCompatible__ThrowsException()
        {
            // Arrange
            var graph = new Graph();

            var var1 = graph.Variable(1, 1, 2, 3);
            var1.Load(new float[] { 0, 0, 0, 0, 0, 0 });

            var var2 = graph.Variable(1, 1, 2, 2);
            var2.Load(new float[] { 1, 2, 3, 4 });

            // Act
            var testDelegate = new TestDelegate(() => { graph.Add(var1, var2); });

            // Assert
            var ex = Assert.Throws<Exception>(testDelegate);
            Assert.That(ex.Message, Is.EqualTo("Tensors are not compatible."));
        }

        [Test]
        public void Forward__DifferentRanks_DifferentShape_TensorsAreNotCompatible__ThrowsException()
        {
            // Arrange
            var graph = new Graph();

            var var1 = graph.Variable(1, 1, 2, 3);
            var1.Load(new float[] { 0, 0, 0, 0, 0, 0 });

            var var2 = graph.Variable(1, 1, 2, 2, 2);
            var2.Load(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 });

            // Act
            var testDelegate = new TestDelegate(() => { graph.Add(var1, var2); });

            // Assert
            var ex = Assert.Throws<Exception>(testDelegate);
            Assert.That(ex.Message, Is.EqualTo("Tensors are not compatible."));
        }
    }
}
