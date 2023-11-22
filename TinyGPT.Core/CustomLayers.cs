using System;
using System.Buffers;
using System.Collections.Generic;
using System.Drawing;
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
		public static Tensor TanhELU(Tensor input)
		{
			using(NewDisposeScope()){
				Tensor y;
				using(Tensor x = input.tanh()){
					y = x.max(input);
				}
				using(y){
					return y.add(1).MoveToOuterDisposeScope();
				}
			}
		}
		
	}
	public interface IL1Regularizable
	{
		public void L1Regularize(double lambda);
	}
	public sealed class ResidualComputeLayer : Module<Tensor, Tensor>, IL1Regularizable
	{
		private readonly Linear input;
		private readonly Linear output;
		private readonly LayerNorm layerNorm;
		private readonly Parameter gate;
		public ResidualComputeLayer(string name, int size, int coresize, double epsilon) : base(name)
		{
			layerNorm = LayerNorm(size, epsilon, false);
			input = Misc.CreateXavierInitializedLinear(size, coresize, true);
			gate = Parameter(ones(1, size));

			output = Misc.CreateXavierInitializedLinear(coresize, size, false);
		}

		public override Tensor forward(Tensor input1)
		{
			using (NewDisposeScope())
			{
				Tensor y;
				using (Tensor x = input.forward(input1))
				{
					y = CustomActivations.TanhELU(x);
				}

				using(Tensor x = y){
					y = output.forward(x);
				}
				using (Tensor x = y)
				{
					using Tensor gated = input1.mul(gate);
					y = x.add(gated);
				}

				using (y)
				{
					return layerNorm.forward(y).MoveToOuterDisposeScope();
				}
			}
		}

		public void L1Regularize(double lambda)
		{
			//NOTE: forget gate is NOT regularized
			Misc.L1RegularizeIMPL(input.weight, lambda);
			Misc.L1RegularizeIMPL(output.weight, lambda);
		}
	}
	public sealed class MultiheadResidualAttention : Module<Tensor, Tensor>, IL1Regularizable
	{
		public void L1Regularize(double lambda)
		{
			Misc.L1RegularizeIMPL(exit.weight, lambda);
			foreach (AttentionLayer attentionLayer in attentionLayers)
			{
				attentionLayer.L1Regularize(lambda);
			}
		}

		public override Tensor forward(Tensor input)
		{
			return Forward(input, input, null);
		}
		public Tensor Forward(Tensor input, Tensor target, Tensor? mask = null)
		{
			ModuleList<AttentionLayer> attentionLayers = this.attentionLayers;
			int heads = attentionLayers.Count;
			Tensor[] tensors = new Tensor[heads];
			using (NewDisposeScope())
			{
				Tensor x;
				using (NewDisposeScope())
				{
					for (int i = 0; i < heads; ++i)
					{
						tensors[i] = attentionLayers[i].Forward(target, input, mask);
					}
					x = cat(tensors, 1).MoveToOuterDisposeScope();
				}

				using (Tensor y = x)
				{
					x = exit.forward(y);
				}
				using (Tensor y = x){
					using Tensor gated = input.mul(gate);
					x = y.add(gated);
				}
				using (x)
				{
					return layerNorm.forward(x).MoveToOuterDisposeScope();
				}
			}


		}

		private readonly Linear exit;
		private readonly LayerNorm layerNorm;
		private readonly ModuleList<AttentionLayer> attentionLayers = new ModuleList<AttentionLayer>();
		private readonly Parameter gate;
		public MultiheadResidualAttention(string name, int inputSize, int keySize, int valueSize, int outputSize, int heads, double epsilon) : base(name)
		{
			for (int i = 0; i < heads; ++i)
			{
				attentionLayers.Add(new AttentionLayer("", inputSize, keySize, valueSize));
			}
			gate = Parameter(ones(1, inputSize));
			exit = Misc.CreateXavierInitializedLinear(valueSize * heads, outputSize, false);
			layerNorm = LayerNorm(outputSize, epsilon, false);
		}
	}
	public sealed class AttentionLayer : Module<Tensor, Tensor>, IL1Regularizable
	{
		private readonly Linear key;
		private readonly Linear value;
		private readonly Linear query;
		public void L1RegularizeValue(double lambda)
		{
			Tensor tensor = value.weight ?? throw new Exception("value does not have weights (should not reach here)");
			Tensor grad = tensor.grad() ?? throw new Exception("Unexpected null gradients (should not reach here)");
			using (NewDisposeScope())
			{
				using Tensor sign = tensor.sign();

				grad.add_(sign, lambda * Math.Sqrt(6.0 / (tensor.size(0) + tensor.size(1))));
			}
		}

		public AttentionLayer(string name, int inputSize, int keySize, int valueSize) : base(name)
		{
			key = Misc.CreateXavierInitializedLinear(inputSize, keySize, false);
			value = Misc.CreateXavierInitializedLinear(inputSize, valueSize, false);
			query = Misc.CreateXavierInitializedLinear(inputSize, keySize, false);
			RegisterComponents();
		}

		public override Tensor forward(Tensor input)
		{
			using (NewDisposeScope())
			{
				using Tensor k = key.forward(input);
				using Tensor q = query.forward(input);
				using Tensor v = value.forward(input);
				return scaled_dot_product_attention(q, k, v, is_casual: true).MoveToOuterDisposeScope();
			}
		}
		public Tensor Forward(Tensor input, Tensor? mask = null)
		{
			using (NewDisposeScope())
			{
				using Tensor k = key.forward(input);
				using Tensor q = query.forward(input);
				using Tensor v = value.forward(input);
				return scaled_dot_product_attention(q, k, v, mask).MoveToOuterDisposeScope();
			}
		}
		public Tensor Forward(Tensor a, Tensor b, Tensor? mask = null)
		{
			using (NewDisposeScope())
			{
				using Tensor k = key.forward(a);
				using Tensor v = value.forward(a);
				using Tensor q = query.forward(b);
				return scaled_dot_product_attention(q, k, v, mask).MoveToOuterDisposeScope();
			}
		}

		public void L1Regularize(double lambda)
		{
			Misc.L1RegularizeIMPL(key.weight, lambda);
			Misc.L1RegularizeIMPL(value.weight, lambda);
			Misc.L1RegularizeIMPL(query.weight, lambda);
		}
	}






}
