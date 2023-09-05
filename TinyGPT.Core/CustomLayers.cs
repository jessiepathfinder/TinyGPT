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
	public sealed class LeakySoftplus : Module<Tensor, Tensor>
	{
		public LeakySoftplus(string name) : base(name)
		{
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			return softplus(input, 1, 32).add(input / 128.0f);
		}
	}
	public sealed class DenseStep : Module<Tensor, Tensor>
	{
		private readonly Module<Tensor, Tensor> network;
		private readonly Module<Tensor, Tensor> activation;
		public DenseStep(long inputs, long outputs, string name) : base(name)
		{
			network = Linear(inputs, outputs);
			activation = new LeakySoftplus("");
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			return activation.forward(network.forward(input));
		}
	}
}
