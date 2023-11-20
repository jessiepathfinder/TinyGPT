using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TinyGPT.Core
{
	public static class Misc
	{
		public static void L1RegularizeIMPL(Tensor? tensor, double lambda){

			if(tensor is null) throw new ArgumentNullException(nameof(tensor));

			//Xavier lambda scaling
			lambda *= Math.Sqrt(6.0 / (tensor.size(0) + tensor.size(1)));
			
			using(no_grad()){
				Tensor grad = tensor.grad() ?? throw new Exception("No gradients to regularize");
				using (NewDisposeScope()){
					using Tensor sign = tensor.sign();
					grad.add_(sign);
				}
			}
			
		}
		public static Linear CreateXavierInitializedLinear(int inputs, int outputs, bool bias){
			Linear linear = Linear(inputs, outputs, bias);
			init.xavier_normal_(linear.weight ?? throw new Exception("No weight found (should not reach here)"));
			return linear;
		}
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

		public static void AdaptiveLearningRateSGD(Tensor parameter, Tensor momentum, double beta, double baseLearningRate){
			Tensor gradient = parameter.grad() ?? throw new Exception("No grad!");

			using(Tensor abs = momentum.abs()){
				abs.mul_(baseLearningRate);
				abs.mul_(gradient);
				parameter.sub_(abs);
			}
			

			momentum.mul_(beta);
			using(Tensor sign = gradient.sign()) {
				momentum.add_(sign, 1 - beta);
			}
		}


	}
}
