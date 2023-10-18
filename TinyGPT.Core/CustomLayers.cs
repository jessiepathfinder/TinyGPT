using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TinyGPT.Core
{
	public static class CustomActivations
	{
		private static readonly long[] dims = new long[] {1};
		public static Tensor LeakySoftplus(Tensor input)
		{
			using Tensor a = input / 16;
			using Tensor b = softplus(input, 1, 20);
			return a.add(b);
		}
		public static Tensor Norm(Tensor input, double epsilon)
		{
			using DisposeScope disposeScope = NewDisposeScope();
			return input.div(input.square().sum(1, true).sqrt().add(epsilon)).MoveToOuterDisposeScope();

		}
		public static Tensor SwishDerivative(Tensor input)
		{
			using DisposeScope disposeScope = NewDisposeScope();
			Tensor sigmoid = input.sigmoid();

			return sigmoid.neg().add(1).mul(input).add(1).mul(sigmoid).MoveToOuterDisposeScope();
		}
	}
	public sealed class JITNetLayer : Module<Tensor, Tensor>
	{
		//In JIT training, some weights are computed at runtime instead of train-time
		//Stolen from GPT-2 (except different activation function)
		private readonly Linear key;
		private readonly Linear query;
		private readonly Linear value;
		private readonly Linear compute1;
		private readonly Linear compute2;


		private readonly double epsilon;

		public JITNetLayer(string name, int inputSize, double epsilon) : base(name)
		{
			key = Linear(inputSize, inputSize, false);
			query = Linear(inputSize, inputSize, false);
			value = Linear(inputSize, inputSize, false);
			compute1 = Linear(inputSize, inputSize);
			compute2 = Linear(inputSize, inputSize);

			this.epsilon = epsilon;
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			using (NewDisposeScope())
			{
				Tensor z;
				{
					using Tensor k = key.forward(input);
					using Tensor v = value.forward(input);
					using Tensor q = query.forward(input);
					z = scaled_dot_product_attention(q, k, v, is_casual: true);
				}

				using (Tensor p = z)
				{
					z = p.add(input);
				}
				using (Tensor p = z)
				{
					z = CustomActivations.Norm(p, epsilon);
				}

				Tensor y;

				using (Tensor p = compute1.forward(z))
				{
					y = CustomActivations.SwishDerivative(p);
				}
				using (Tensor p = y)
				{
					y = compute2.forward(p);
				}
				using (Tensor p = y)
				{
					using (z){
						y = p.add(z);
					}
				}
				

				return CustomActivations.Norm(y, epsilon).MoveToOuterDisposeScope();
			}
		}
	}


}
