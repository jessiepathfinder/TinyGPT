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
		public static Tensor SignRoot(Tensor input) {
			using DisposeScope disposeScope = NewDisposeScope();
			Tensor sign = input.tanh();
			Tensor res = sign.mul(input).add(1).sqrt().mul(sign);
			res.MoveToOuterDisposeScope();
			return res;
		}
	}


	public sealed class JessieNetLayer : Module<Tensor, Tensor>
	{
		private readonly Linear a1;
		private readonly Linear a2;
		private readonly PReLU relu;
		public JessieNetLayer(string name, int inputSize, int hiddenSize) : base(name)
		{
			a1 = Linear(inputSize, hiddenSize);
			a2 = Linear(hiddenSize, inputSize);
			relu = PReLU(inputSize, 1);
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			using DisposeScope disposeScope = NewDisposeScope();
			Tensor res = relu.forward(a2.forward(CustomActivations.SignRoot(a1.forward(input))).add(input));

			res.MoveToOuterDisposeScope();
			return res;
		}
	}


}
