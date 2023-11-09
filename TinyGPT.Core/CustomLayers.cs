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
		public static Tensor LeakySoftplus(Tensor input)
		{
			using Tensor a = input / 16;
			using Tensor b = softplus(input, 1, 20);
			return a.add(b);
		}
		public static Tensor SwishDerivative(Tensor input)
		{
			using DisposeScope disposeScope = NewDisposeScope();
			Tensor sigmoid = input.sigmoid();

			Tensor y;
			using (Tensor x = sigmoid.neg())
			{
				y = x.add(1);
			}
			using (Tensor x = y)
			{
				y = x.mul(input);
			}
			using (Tensor x = y)
			{
				y = x.mul(sigmoid);
			}
			using (Tensor x = y)
			{
				y = x.add(sigmoid);
			}

			return y.MoveToOuterDisposeScope();
		}
	}
	public sealed class AttentionLayer : Module<Tensor, Tensor>
	{
		private readonly Linear key;
		private readonly Linear value;
		private readonly Linear query;


		public AttentionLayer(string name, int inputSize) : base(name)
		{
			key = Misc.CreateXavierInitializedLinear(inputSize, inputSize, false);
			value = Misc.CreateXavierInitializedLinear(inputSize, inputSize, false);
			query = Misc.CreateXavierInitializedLinear(inputSize, inputSize, false);
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			using (NewDisposeScope())
			{
				Tensor r;
				using (Tensor x2 = input.unsqueeze(0))
				{
					using Tensor k = key.forward(x2);
					using Tensor q = query.forward(x2);
					using Tensor v = value.forward(x2);
					r = scaled_dot_product_attention(q, k, v, is_casual: true);
				}


				using (r)
				{
					return r.squeeze(0).MoveToOuterDisposeScope();
				}
			}
		}
	}

	




}
