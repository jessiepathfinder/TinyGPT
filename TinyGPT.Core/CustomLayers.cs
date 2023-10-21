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
			using(Tensor x = sigmoid.neg()){
				y = x.add(1);
			}
			using(Tensor x = y){
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
	public sealed class ResidualAttentionLayer : Module<Tensor, Tensor>
	{
		private readonly Linear key;
		private readonly Linear value;
		private readonly Linear query;
		private readonly LayerNorm norm;


		public ResidualAttentionLayer(string name, int inputSize, double epsilon) : base(name)
		{
			key = Linear(inputSize, inputSize, false);
			value = Linear(inputSize, inputSize, false);
			query = Linear(inputSize, inputSize, false);
			norm = LayerNorm(inputSize, epsilon);
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			using (NewDisposeScope()) {
				Tensor r;
				{
					using Tensor k = key.forward(input);
					using Tensor q = value.forward(input);
					using Tensor v = query.forward(input);
					r = scaled_dot_product_attention(q, k, v, is_casual: true);
				}
				using(Tensor x = r){
					r = r.add(input);
				}
				return norm.forward(r).MoveToOuterDisposeScope();
			}
		}
	}

	public sealed class ResidualComputeLayer : Module<Tensor, Tensor>
	{
		private readonly Linear compute1;
		private readonly Linear compute2;
		private readonly LayerNorm norm;


		public ResidualComputeLayer(string name, int inputSize, double epsilon) : base(name)
		{
			compute1 = Linear(inputSize, inputSize);
			compute2 = Linear(inputSize, inputSize);
			norm = LayerNorm(inputSize, epsilon);
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			using (NewDisposeScope())
			{
				Tensor r;
				using (Tensor x = compute1.forward(input))
				{
					r = CustomActivations.SwishDerivative(x);
				}
				using(Tensor x = r){
					r = compute2.forward(x);
				}
				using (Tensor x = r)
				{
					r = x.add(input);
				}
				using(r){
					return norm.forward(r).MoveToOuterDisposeScope();
				}
			}
		}
	}

	public sealed class AttentionBlock : Module<Tensor, Tensor>
	{
		private readonly ResidualAttentionLayer attention;
		private readonly ResidualComputeLayer compute;

		public AttentionBlock(string name, int inputSize, double epsilon) : base(name)
		{
			attention = new ResidualAttentionLayer("", inputSize, epsilon);
			compute = new ResidualComputeLayer("", inputSize, epsilon);
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			using (NewDisposeScope())
			{
				Tensor z;
				using(Tensor x = attention.forward(input)){
					z = compute.forward(x);
				}
				return z.MoveToOuterDisposeScope();
			}
		}
	}


}
