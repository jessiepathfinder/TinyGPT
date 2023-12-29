using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TinyGPT.Core
{
	public static class Misc
	{
		private static readonly NLLLoss nlloss = new NLLLoss(reduction: Reduction.None);
		public static Tensor FastCrossEntropyLoss(Tensor input, Tensor logits, double squareboost, bool average, Tensor? boost = null) {
			using(NewDisposeScope()){
				Tensor z;
				using(Tensor y = input.exp()){
					z = y.sum(-1, false);
				}
				using(Tensor y = z){
					z = y.log();
				}
				Tensor x;
				using(Tensor y = nlloss.forward(input, logits))
				{
					using(z){
						x = z.add(y);
					}
				}
				if (boost is { })
				{
					using Tensor y = x;
					x = y.mul(boost);
				}
				if (squareboost > 0)
				{
					using Tensor y = x;
					using Tensor y2 = x.mul(x);
					x = y.add(y2, squareboost);
				}
				if (average)
				{
					using Tensor y = x;
					x = y.mean();
				}
				else
				{
					using Tensor y = x;
					x = y.sum();
				}
				return x.MoveToOuterDisposeScope();


			}
		}
		

		public static void L2RegularizeIMPL(Tensor? tensor, Scalar lambda)
		{

			if (tensor is null) throw new ArgumentNullException(nameof(tensor));

			(tensor.grad() ?? throw new Exception("No gradients to regularize")).add_(tensor, lambda);

		}
		public static Linear CreateXavierInitializedLinear(int inputs, int outputs, bool bias, double gain = 1.0)
		{
			Linear linear = Linear(inputs, outputs, bias);
			init.xavier_normal_(linear.weight ?? throw new Exception("No weight found (should not reach here)"), gain);
			return linear;
		}
		public static Linear CreateKaimingInitializedLinear(int inputs, int outputs, bool bias, init.FanInOut fanmode, double gain = 1.0)
		{
			//temptest
			//return CreateXavierInitializedLinear(inputs, outputs, bias, gain);
			Linear linear = Linear(inputs, outputs, bias);
			init.kaiming_normal_(linear.weight ?? throw new Exception("No weight found (should not reach here)"), gain, fanmode);
			return linear;
		}
		public static void EraseReturnAsync<T>(ArrayPool<T> arrayPool, T[] array, int erase) where T : class
		{
			ThreadPool.QueueUserWorkItem(EraseReturn, (arrayPool, array, erase), true);
		}
		private static void EraseReturn<T>((ArrayPool<T> arrayPool, T[] array, int erase) x) where T : class
		{
			for (int i = 0; i < x.erase; ++i)
			{
#pragma warning disable CS8625 // Cannot convert null literal to non-nullable reference type.
				x.array[i] = null;
#pragma warning restore CS8625 // Cannot convert null literal to non-nullable reference type.
			}
			x.arrayPool.Return(x.array);
		}

		public static void AdaptiveLearningRateSGD(Tensor parameter, Tensor momentum, double beta, double baseLearningRate)
		{
			Tensor gradient = parameter.grad() ?? throw new Exception("No grad!");

			using (Tensor abs = momentum.abs())
			{
				abs.mul_(baseLearningRate);
				abs.mul_(gradient);
				parameter.sub_(abs);
			}


			momentum.mul_(beta);
			using (Tensor sign = gradient.sign())
			{
				momentum.add_(sign, 1 - beta);
			}
		}
		public static Tensor MixedPrecisionAttention(Tensor query, Tensor key, Tensor value, Tensor? mask = null, bool causal = false)
		{
			ScalarType scalarType = query.dtype;
			if (scalarType == ScalarType.Float64)
			{
				return scaled_dot_product_attention(query, key, value, mask, 0, causal);
			}
			using (NewDisposeScope())
			{
				Tensor x;
				{
					using Tensor q = query.to(ScalarType.Float64);
					using Tensor k = key.to(ScalarType.Float64);
					using Tensor v = value.to(ScalarType.Float64);
					x = scaled_dot_product_attention(q, k, v, mask, 0, causal);
				}
				using (x)
				{
					return x.to(scalarType).MoveToOuterDisposeScope();
				}
			}
		}
		public static Tensor GenerateXavierQueryMatrix(int inputs, int outputs, int heads, ScalarType? scalarType = null, Device? device = null, bool require_grad = false){
			Span<long> sizes = stackalloc long[3];
			sizes[0] = heads;
			sizes[1] = inputs;
			sizes[2] = outputs;
			return normal(0, Math.Sqrt(2.0 / (inputs + (outputs * heads))), sizes, scalarType, device, require_grad);
		}

	}
}
