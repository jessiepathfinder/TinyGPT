using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace TinyGPT.Core
{
	public static class Misc
	{
		public static void EraseReturnAsync<T>(ArrayPool<T> arrayPool, T[] array, int erase) where T : class
		{
			ThreadPool.QueueUserWorkItem(EraseReturn, (arrayPool, array, erase), true);
		}
		private static void EraseReturn<T>((ArrayPool<T> arrayPool, T[] array, int erase) x) where T : class{
			for(int i = 0; i < x.erase; ++i){
#pragma warning disable CS8625 // Cannot convert null literal to non-nullable reference type.
				x.array[i] = null;
#pragma warning restore CS8625 // Cannot convert null literal to non-nullable reference type.
			}
			x.arrayPool.Return(x.array);
		}

		public static Tensor ComputeSoftmaxLoss2(Tensor x, int expectedclass){
			Tensor squared = x.square();

			return squared.sum().subtract(squared[expectedclass]).sqrt();
		}
	}
}
