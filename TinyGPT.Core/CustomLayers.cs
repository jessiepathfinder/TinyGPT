using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
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
}
