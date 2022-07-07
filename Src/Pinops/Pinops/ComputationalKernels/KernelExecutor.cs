using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;

namespace Pinops.Core.ComputationalKernels
{
    public sealed class KernelExecutor
    {
        private static readonly Context context = Context.CreateDefault();
        private static readonly ConcurrentDictionary<MethodInfo, Delegate> kernelDelegateMap = new ConcurrentDictionary<MethodInfo, Delegate>();

        internal static Accelerator Accelerator { get; }

        static KernelExecutor()
        {
            Accelerator = context.Devices.First(a => a.AcceleratorType == AcceleratorType.CPU).CreateAccelerator(context);

            Console.WriteLine("Accelerator info:");
            Console.WriteLine($"  Accelerator type: {Accelerator.AcceleratorType}");
            Console.WriteLine($"  Max number of threads: {Accelerator.MaxNumThreads}");
            Console.WriteLine();
        }

        private static void ConsoleWriteLine_KernelAddedToCache(MethodInfo methodInfo)
        {
            Console.WriteLine($"Added {methodInfo.Name} to kernel cache");
        }

        private static void ConsoleWriteLine_KernelCalled(MethodInfo methodInfo, string callerMethodName)
        {
            //Console.WriteLine($"{methodInfo.Name} called from {callerMethodName}");
        }

        private static Action<TIndex, T1, T2> GetOrAddKernel<TIndex, T1, T2>(Action<TIndex, T1, T2> action)
            where TIndex : struct, IIndex
            where T1 : struct
            where T2 : struct
        {
            var methodInfo = action.Method;
            return (Action<TIndex, T1, T2>)kernelDelegateMap.GetOrAdd(
                methodInfo,
                x =>
                {
                    ConsoleWriteLine_KernelAddedToCache(methodInfo);
                    return Accelerator.LoadAutoGroupedStreamKernel(action);
                });
        }

        private static Action<TIndex, T1, T2, T3> GetOrAddKernel<TIndex, T1, T2, T3>(Action<TIndex, T1, T2, T3> action)
            where TIndex : struct, IIndex
            where T1 : struct
            where T2 : struct
            where T3 : struct
        {
            var methodInfo = action.Method;
            return (Action<TIndex, T1, T2, T3>)kernelDelegateMap.GetOrAdd(
                methodInfo,
                x =>
                {
                    ConsoleWriteLine_KernelAddedToCache(methodInfo);
                    return Accelerator.LoadAutoGroupedStreamKernel(action);
                });
        }

        private static Action<TIndex, T1, T2, T3, T4> GetOrAddKernel<TIndex, T1, T2, T3, T4>(Action<TIndex, T1, T2, T3, T4> action)
            where TIndex : struct, IIndex
            where T1 : struct
            where T2 : struct
            where T3 : struct
            where T4 : struct
        {
            var methodInfo = action.Method;
            return (Action<TIndex, T1, T2, T3, T4>)kernelDelegateMap.GetOrAdd(
                methodInfo,
                x =>
                {
                    ConsoleWriteLine_KernelAddedToCache(methodInfo);
                    return Accelerator.LoadAutoGroupedStreamKernel(action);
                });
        }

        private static Action<TIndex, T1, T2, T3, T4, T5> GetOrAddKernel<TIndex, T1, T2, T3, T4, T5>(Action<TIndex, T1, T2, T3, T4, T5> action)
            where TIndex : struct, IIndex
            where T1 : struct
            where T2 : struct
            where T3 : struct
            where T4 : struct
            where T5 : struct
        {
            var methodInfo = action.Method;
            return (Action<TIndex, T1, T2, T3, T4, T5>)kernelDelegateMap.GetOrAdd(
                methodInfo,
                x =>
                {
                    ConsoleWriteLine_KernelAddedToCache(methodInfo);
                    return Accelerator.LoadAutoGroupedStreamKernel(action);
                });
        }

        private static Action<TIndex, T1, T2, T3, T4, T5, T6> GetOrAddKernel<TIndex, T1, T2, T3, T4, T5, T6>(Action<TIndex, T1, T2, T3, T4, T5, T6> action)
            where TIndex : struct, IIndex
            where T1 : struct
            where T2 : struct
            where T3 : struct
            where T4 : struct
            where T5 : struct
            where T6 : struct
        {
            var methodInfo = action.Method;
            return (Action<TIndex, T1, T2, T3, T4, T5, T6>)kernelDelegateMap.GetOrAdd(
                methodInfo,
                x =>
                {
                    ConsoleWriteLine_KernelAddedToCache(methodInfo);
                    return Accelerator.LoadAutoGroupedStreamKernel(action);
                });
        }

        private static Action<TIndex, T1, T2, T3, T4, T5, T6, T7> GetOrAddKernel<TIndex, T1, T2, T3, T4, T5, T6, T7>(Action<TIndex, T1, T2, T3, T4, T5, T6, T7> action)
            where TIndex : struct, IIndex
            where T1 : struct
            where T2 : struct
            where T3 : struct
            where T4 : struct
            where T5 : struct
            where T6 : struct
            where T7 : struct
        {
            var methodInfo = action.Method;
            return (Action<TIndex, T1, T2, T3, T4, T5, T6, T7>)kernelDelegateMap.GetOrAdd(
                methodInfo,
                x =>
                {
                    ConsoleWriteLine_KernelAddedToCache(methodInfo);
                    return Accelerator.LoadAutoGroupedStreamKernel(action);
                });
        }

        private static Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8> GetOrAddKernel<TIndex, T1, T2, T3, T4, T5, T6, T7, T8>(Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8> action)
            where TIndex : struct, IIndex
            where T1 : struct
            where T2 : struct
            where T3 : struct
            where T4 : struct
            where T5 : struct
            where T6 : struct
            where T7 : struct
            where T8 : struct
        {
            var methodInfo = action.Method;
            return (Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8>)kernelDelegateMap.GetOrAdd(
                methodInfo,
                x =>
                {
                    ConsoleWriteLine_KernelAddedToCache(methodInfo);
                    return Accelerator.LoadAutoGroupedStreamKernel(action);
                });
        }

        private static Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9> GetOrAddKernel<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9>(Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9> action)
            where TIndex : struct, IIndex
            where T1 : struct
            where T2 : struct
            where T3 : struct
            where T4 : struct
            where T5 : struct
            where T6 : struct
            where T7 : struct
            where T8 : struct
            where T9 : struct
        {
            var methodInfo = action.Method;
            return (Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9>)kernelDelegateMap.GetOrAdd(
                methodInfo,
                x =>
                {
                    ConsoleWriteLine_KernelAddedToCache(methodInfo);
                    return Accelerator.LoadAutoGroupedStreamKernel(action);
                });
        }

        private static Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> GetOrAddKernel<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> action)
            where TIndex : struct, IIndex
            where T1 : struct
            where T2 : struct
            where T3 : struct
            where T4 : struct
            where T5 : struct
            where T6 : struct
            where T7 : struct
            where T8 : struct
            where T9 : struct
            where T10 : struct
        {
            var methodInfo = action.Method;
            return (Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>)kernelDelegateMap.GetOrAdd(
                methodInfo,
                x =>
                {
                    ConsoleWriteLine_KernelAddedToCache(methodInfo);
                    return Accelerator.LoadAutoGroupedStreamKernel(action);
                });
        }

        private static Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> GetOrAddKernel<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>(Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> action)
           where TIndex : struct, IIndex
           where T1 : struct
           where T2 : struct
           where T3 : struct
           where T4 : struct
           where T5 : struct
           where T6 : struct
           where T7 : struct
           where T8 : struct
           where T9 : struct
           where T10 : struct
           where T11 : struct
        {
            var methodInfo = action.Method;
            return (Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>)kernelDelegateMap.GetOrAdd(
                methodInfo,
                x =>
                {
                    ConsoleWriteLine_KernelAddedToCache(methodInfo);
                    return Accelerator.LoadAutoGroupedStreamKernel(action);
                });
        }

        private static Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12> GetOrAddKernel<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>(Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12> action)
           where TIndex : struct, IIndex
           where T1 : struct
           where T2 : struct
           where T3 : struct
           where T4 : struct
           where T5 : struct
           where T6 : struct
           where T7 : struct
           where T8 : struct
           where T9 : struct
           where T10 : struct
           where T11 : struct
           where T12 : struct
        {
            var methodInfo = action.Method;
            return (Action<TIndex, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>)kernelDelegateMap.GetOrAdd(
                methodInfo,
                x =>
                {
                    ConsoleWriteLine_KernelAddedToCache(methodInfo);
                    return Accelerator.LoadAutoGroupedStreamKernel(action);
                });
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<float>> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<float>> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3) =>
            {
                kernel(T1, T2, T3);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<float>, float> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<float>, float> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4) =>
            {
                kernel(T1, T2, T3, T4);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4) =>
            {
                kernel(T1, T2, T3, T4);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, float> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, float> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5) =>
            {
                kernel(T1, T2, T3, T4, T5);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, float, ArrayView<float>, ArrayView<float>> Execute(
            Action<Index1D, float, ArrayView<float>, ArrayView<float>> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4) =>
            {
                kernel(T1, T2, T3, T4);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, int, int, ArrayView<float>, ArrayView<float>, ArrayView<float>> Execute(
            Action<Index1D, int, int, ArrayView<float>, ArrayView<float>, ArrayView<float>> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6) =>
            {
                kernel(T1, T2, T3, T4, T5, T6);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index2D, int, int, ArrayView<float>, ArrayView<float>, ArrayView<float>> Execute(
            Action<Index2D, int, int, ArrayView<float>, ArrayView<float>, ArrayView<float>> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6) =>
            {
                kernel(T1, T2, T3, T4, T5, T6);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index2D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>> Execute(
            Action<Index2D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5) =>
            {
                kernel(T1, T2, T3, T4, T5);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index2D, ArrayView<float>, int, int, ArrayView<float>, int, int, ArrayView<float>, int, int> Execute(
            Action<Index2D, ArrayView<float>, int, int, ArrayView<float>, int, int, ArrayView<float>, int, int> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) =>
            {
                kernel(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index2D, ArrayView<float>, int, int, ArrayView<float>, int, int, ArrayView<float>, int, int, int, int, int> Execute(
            Action<Index2D, ArrayView<float>, int, int, ArrayView<float>, int, int, ArrayView<float>, int, int, int, int, int> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13) =>
            {
                kernel(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, int, int> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, int, int> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6) =>
            {
                kernel(T1, T2, T3, T4, T5, T6);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, int> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, int> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5) =>
            {
                kernel(T1, T2, T3, T4, T5);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4) =>
            {
                kernel(T1, T2, T3, T4);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<float>> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<float>> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5) =>
            {
                kernel(T1, T2, T3, T4, T5);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, float, float, float, float, float, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>> Execute(
            Action<Index1D, float, float, float, float, float, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) =>
            {
                kernel(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<float>, int, int, int> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<float>, int, int, int> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6) =>
            {
                kernel(T1, T2, T3, T4, T5, T6);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6, T7) =>
            {
                kernel(T1, T2, T3, T4, T5, T6, T7);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int, int, int> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int, int, int> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6, T7, T8) =>
            {
                kernel(T1, T2, T3, T4, T5, T6, T7, T8);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int, int, int> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int, int, int> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10) =>
            {
                kernel(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int, int, int> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int, int, int> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12) =>
            {
                kernel(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int, int, int> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int, int, int> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6, T7, T8, T9) =>
            {
                kernel(T1, T2, T3, T4, T5, T6, T7, T8, T9);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int, int> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, int, int> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6, T7, T8) =>
            {
                kernel(T1, T2, T3, T4, T5, T6, T7, T8);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6, T7) =>
            {
                kernel(T1, T2, T3, T4, T5, T6, T7);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6, T7, T8, T9) =>
            {
                kernel(T1, T2, T3, T4, T5, T6, T7, T8, T9);
                Accelerator.Synchronize();
            };
        }

        internal static Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<int>> Execute(
            Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<int>> action,
            [CallerMemberName] string callerMethodName = "")
        {
            ConsoleWriteLine_KernelCalled(action.Method, callerMethodName);

            var kernel = GetOrAddKernel(action);
            return (T1, T2, T3, T4, T5, T6, T7) =>
            {
                kernel(T1, T2, T3, T4, T5, T6, T7);
                Accelerator.Synchronize();
            };
        }
    }
}
