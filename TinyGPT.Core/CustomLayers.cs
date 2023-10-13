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
		public static Tensor SwishDerivative(Tensor input) {
			using DisposeScope disposeScope = NewDisposeScope();
			Tensor sigmoid = input.sigmoid();
			
			return sigmoid.neg().add(1).mul(input).add(1).mul(sigmoid).MoveToOuterDisposeScope();
		}
		public static Tensor Norm(Tensor input, double epsilon){
			using DisposeScope disposeScope = NewDisposeScope();
			return input.div(input.square().mean().sqrt().add(epsilon)).MoveToOuterDisposeScope();

		}
	}


	public sealed class JessieNetLayer : Module<Tensor, Tensor>
	{
		private readonly Linear a1;
		private readonly Linear a2;
		private readonly double epsilon;

		public JessieNetLayer(string name, int inputSize, int hiddenSize, double epsilon) : base(name)
		{
			a1 = Linear(inputSize, hiddenSize);
			a2 = Linear(hiddenSize, inputSize);
			this.epsilon = epsilon;
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			using DisposeScope disposeScope = NewDisposeScope();
			Tensor res = CustomActivations.Norm(a2.forward(CustomActivations.SwishDerivative(a1.forward(input))).add(input), epsilon);
			

			res.MoveToOuterDisposeScope();
			return res;
		}
	}
	public sealed class JITNetLayer : Module<Tensor, Tensor>
	{
		//In JIT training, some weights are computed at runtime instead of train-time
		private readonly Linear key;
		private readonly Linear query;
		private readonly Linear value;
		private readonly Linear compute;

		private readonly Parameter bias;



		private readonly double epsilon;

		public JITNetLayer(string name, int inputSize, double epsilon) : base(name)
		{
			key = Linear(inputSize, inputSize, false);
			query = Linear(inputSize, inputSize, false);
			value = Linear(inputSize, inputSize, false);
			compute = Linear(inputSize, inputSize);
			bias = new Parameter(zeros(1, inputSize));

			this.epsilon = epsilon;
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			using(NewDisposeScope()){
				Tensor z;
				{
					using Tensor k = key.forward(input);
					using Tensor v = value.forward(input);
					using Tensor q = query.forward(input);
					z = scaled_dot_product_attention(q, k, v, is_casual: true);
				}
				using (Tensor p = z)
				{
					z = p.add(bias);
				}
				using (Tensor p = z)
				{
					z = CustomActivations.SwishDerivative(p);
				}
				using(Tensor p = z){
					z = compute.forward(p);
				}
				using (Tensor p = z)
				{
					z = p.add(input);
				}
				using (Tensor p = z)
				{
					z = CustomActivations.Norm(p, epsilon);
				}
				return z.MoveToOuterDisposeScope();
			}
		}
	}


}
