﻿using System;
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
	}


	public sealed class JessieNetLayer : Module<Tensor, Tensor>
	{
		private readonly Linear a1;
		private readonly Linear a2;
		private readonly PReLU relu;
		private readonly Parameter normbias;
		private readonly double epsilon;

		public JessieNetLayer(string name, int inputSize, int hiddenSize, double epsilon = 1e-8) : base(name)
		{
			a1 = Linear(inputSize, hiddenSize);
			a2 = Linear(hiddenSize, inputSize);
			relu = PReLU(inputSize, 1);
			normbias = new Parameter(zeros(inputSize));
			this.epsilon = epsilon;
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			using DisposeScope disposeScope = NewDisposeScope();
			Tensor res = relu.forward(a2.forward(CustomActivations.SwishDerivative(a1.forward(input)))).add(input);
			res = res.div(res.add(normbias).square().mean().sqrt().add(epsilon));
			

			res.MoveToOuterDisposeScope();
			return res;
		}
	}


}
