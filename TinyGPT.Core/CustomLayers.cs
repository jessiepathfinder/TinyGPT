using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
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
			return softplus(input, 1, 32).add(input / 16.0f);
		}
	}
	public sealed class DenseStep : Module<Tensor, Tensor>
	{
		private readonly Module<Tensor, Tensor> network;
		public DenseStep(long inputs, long outputs, string name) : base(name)
		{
			network = Linear(inputs, outputs);
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			return CustomActivations.LeakySoftplus(network.forward(input));
		}
	}
	public sealed class DenseStepV2 : Module<Tensor, Tensor>
	{
		private readonly Module<Tensor, Tensor> network;
		private readonly PReLU relu;
		public DenseStepV2(long inputs, long outputs, string name) : base(name)
		{
			network = Linear(inputs, outputs);
			relu = PReLU(outputs);
			RegisterComponents();
		}
		public DenseStepV2(long inputs, long outputs, bool hasbias, string name) : base(name)
		{
			network = Linear(inputs, outputs, hasbias);
			relu = PReLU(outputs);
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			return relu.forward(network.forward(input));
		}
	}
	public class Tridentv2 : Module<Tensor, (Tensor, Tensor, Tensor)>
	{
		private readonly Linear linear1;
		private readonly Linear linear2;
		private readonly Linear linear3;

		private readonly PReLU prelu1;
		private readonly PReLU prelu2;
		private readonly PReLU prelu3;

		public Tridentv2(string name, int insize, int outsize) : base(name)
		{
			linear1 = Linear(insize, outsize, false);
			linear2 = Linear(insize, outsize, false);
			linear3 = Linear(insize, outsize, false);

			prelu1 = PReLU(outsize, 1);
			prelu2 = PReLU(outsize, 1);
			prelu3 = PReLU(outsize, 1);
			RegisterComponents();
		}

		public override (Tensor, Tensor, Tensor) forward(Tensor input1)
		{
			return (prelu1.forward(linear1.forward(input1)), prelu2.forward(linear2.forward(input1)), prelu3.forward(linear3.forward(input1)));
		}
		public Tensor ValueOnly(Tensor input)
		{
			return linear3.forward(input);
		}
	}
}
