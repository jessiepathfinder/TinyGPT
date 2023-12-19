using System;
using System.Buffers;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
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
			using (NewDisposeScope())
			{
				using Tensor x = input.tanh();
				return x.max(input).MoveToOuterDisposeScope();
			}
		}

	}
	public interface IL2Regularizable
	{
		public void L2Regularize(double lambda);
	}
	public sealed class AutogatedSwish : Module<Tensor, Tensor>{
		private readonly Parameter parameter;
		private readonly Parameter makeupGain;

		public AutogatedSwish(string name, double initialParameter, double initialMakeupGain, int size) : base(name)
		{
			parameter = Parameter(zeros(size).fill_(initialParameter));
			makeupGain = Parameter(zeros(size).fill_(initialMakeupGain));
		}

		public override Tensor forward(Tensor input)
		{
			using(NewDisposeScope()){
				Tensor y;
				using(Tensor x = input.mul(parameter)){
					y = x.sigmoid();
				}
				using (Tensor x = y)
				{
					y = x.mul(input);
				}
				using (y){
					return y.mul(makeupGain).MoveToOuterDisposeScope();
				}
			}
		}
	}
	public sealed class ResidualAutogatedComputeLayer : Module<Tensor, Tensor>, IL2Regularizable
	{
		private readonly Linear input;
		private readonly Linear output;
		private readonly AutogatedSwish finalActivation;
		private readonly LayerNorm layerNorm;
		public ResidualAutogatedComputeLayer(string name, int size, int coresize, double epsilon) : base(name)
		{
			layerNorm = LayerNorm(size, epsilon, false);
			input = Misc.CreateXavierInitializedLinear(size, coresize, true);
			finalActivation = new AutogatedSwish("", 0, 2, size);

			output = Misc.CreateXavierInitializedLinear(coresize, size, true);
			RegisterComponents();
		}

		public override Tensor forward(Tensor input1)
		{
			using (NewDisposeScope())
			{
				Tensor y;
				using (Tensor x = input.forward(input1))
				{
					y = x.gelu();
				}
				using(Tensor x = y){
					y = output.forward(x);
				}
				using (Tensor x = y)
				{
					y = finalActivation.forward(x);
				}
				using (y)
				{
					return layerNorm.forward(y).MoveToOuterDisposeScope();
				}
			}
		}

		public void L2Regularize(double lambda)
		{
			//NOTE: gate doesn't need regularization
			Misc.L2RegularizeIMPL(input.weight, lambda);
			Misc.L2RegularizeIMPL(output.weight, lambda);
		}
	}
	public sealed class ResidualAutogatedMultiQueryAttention : Module<Tensor, Tensor>
	{
		public override Tensor forward(Tensor input)
		{
			return Forward(input, input, null);
		}
		public Tensor Forward(Tensor input, Tensor target, Tensor? mask = null)
		{
			using (NewDisposeScope())
			{
				Tensor x;
				using (Tensor q = input.matmul(queries), k = keys.forward(target), v = values.forward(target))
				{
					x = Misc.MixedPrecisionAttention(q, k, v, mask, false);
				}
				using (Tensor y = x)
				{
					x = y.transpose(0, 1);
				}
				using (Tensor y = x)
				{
					x = y.flatten(1);
				}
				using (Tensor y = x)
				{
					x = exit.forward(y);
				}
				using (Tensor y = x)
				{
					x = finalActivation.forward(y);
				}
				using (x)
				{
					return layerNorm.forward(x).MoveToOuterDisposeScope();
				}
			}


		}

		private readonly Linear exit;
		private readonly LayerNorm layerNorm;
		private readonly Parameter queries;
		private readonly Linear keys;
		private readonly Linear values;
		private readonly AutogatedSwish finalActivation;

		public ResidualAutogatedMultiQueryAttention(string name, int inputSize, int keySize, int valueSize, int heads, double epsilon) : base(name)
		{
			queries = Parameter(Misc.GenerateXavierQueryMatrix(inputSize, keySize, heads));
			finalActivation = new AutogatedSwish("", 0, 2, inputSize);
			exit = Misc.CreateXavierInitializedLinear(valueSize * heads, inputSize, true);
			layerNorm = LayerNorm(inputSize, epsilon, false);
			keys = Misc.CreateXavierInitializedLinear(inputSize, keySize, false);
			values = Misc.CreateXavierInitializedLinear(inputSize, valueSize, false);
			RegisterComponents();
		}
	}







}
