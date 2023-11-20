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
	public interface IL1Regularizable{
		public void L1Regularize(double lambda);
	}
	public sealed class ResidualGatedComputeLayer : Module<Tensor, Tensor>, IL1Regularizable
	{
		private readonly Linear input;
		private readonly Linear output;
		private readonly Linear gate;
		private readonly LayerNorm layerNorm;
		private static readonly Scalar one = 1;
		public ResidualGatedComputeLayer(string name, int size, double epsilon) : base(name)
		{
			layerNorm = LayerNorm(size, epsilon, false);
			input = Misc.CreateXavierInitializedLinear(size, size, true);
			gate = Misc.CreateXavierInitializedLinear(size, size, true);

			output = Misc.CreateXavierInitializedLinear(size, size, false);
		}

		public override Tensor forward(Tensor input1)
		{
			using(NewDisposeScope()){
				Tensor core;
				using(Tensor x = input.forward(input1)){
					core = x.gelu();
				}
				Tensor y;

				using(Tensor x = gate.forward(input1)){
					y = x.sigmoid();
				}
				Tensor computed;
				using (core)
				{
					computed = output.forward(core);
				}

				Tensor output3;
				using (y){
					Tensor output2;
					using(computed){
						using Tensor flip = one - y;
						output2 = computed.mul(flip);
					}


					using Tensor residual = input1.mul(y);
					using (output2)
					{
						output3 = residual.add(output2);
					}

				}
				using (output3)
				{
					return layerNorm.forward(output3).MoveToOuterDisposeScope();
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
			foreach(AttentionLayer attentionLayer in attentionLayers) {
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
				using (x)
				{
					return layerNorm.forward(x).MoveToOuterDisposeScope();
				}
			}


		}

		private readonly Linear exit;
		private readonly LayerNorm layerNorm;
		private readonly ModuleList<AttentionLayer> attentionLayers = new ModuleList<AttentionLayer>();
		public MultiheadResidualAttention(string name, int inputSize, int keySize, int valueSize, int outputSize, int heads, double epsilon) : base(name)
		{
			for(int i = 0; i < heads; ++i){
				attentionLayers.Add(new AttentionLayer("", inputSize, keySize, valueSize));
			}
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
