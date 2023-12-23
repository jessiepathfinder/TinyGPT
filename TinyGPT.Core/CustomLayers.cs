using System;
using System.Buffers;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
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
		public static Tensor KernelNorm(Tensor input, long kernelSize, double epsilon){
			Tensor z;
			Tensor std;
			using (NewDisposeScope())
			{
				using (Tensor x = input.unflatten(1, input.size(1) / kernelSize, kernelSize))
				{
					(std, Tensor mean) = x.std_mean(2, false, true);
					using (mean)
					{
						z = x.sub(mean);
					}
				}
				
				using (Tensor x = std)
				{
					std = x.add(epsilon);
				}
				using (Tensor x = z)
				{
					using (std)
					{
						z = x.div(std);
					}
				}
				using (z)
				{
					return z.flatten(1, 2).MoveToOuterDisposeScope();
				}
			}
		}
	}
	public interface IL2Regularizable
	{
		public void L2Regularize(double lambda);
	}
	public sealed class ResidualComputeLayer : Module<Tensor, Tensor>, IL2Regularizable
	{
		private readonly Linear input;
		private readonly Linear output;
		private readonly long normKernelSize;
		private readonly double epsilon;
		public ResidualComputeLayer(string name, int size, int coresize, double epsilon, int normKernelSize) : base(name)
		{
			input = Misc.CreateXavierInitializedLinear(size, coresize, true);
			output = Misc.CreateXavierInitializedLinear(coresize, size, true);
			this.normKernelSize = normKernelSize;
			this.epsilon = epsilon;
			RegisterComponents();
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
				using (y)
				{
					return CustomActivations.KernelNorm(y, normKernelSize, epsilon).MoveToOuterDisposeScope();
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
	public sealed class ResidualMultiQueryAttention : Module<Tensor, Tensor>
	{
		private readonly long normKernelSize;
		private readonly double epsilon;
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
				using (x)
				{
					return CustomActivations.KernelNorm(x, normKernelSize, epsilon).MoveToOuterDisposeScope();
				}
			}


		}

		private readonly Linear exit;
		private readonly Parameter queries;
		private readonly Linear keys;
		private readonly Linear values;

		public ResidualMultiQueryAttention(string name, int inputSize, int keySize, int valueSize, int heads, double epsilon, long normKernelSize) : base(name)
		{
			queries = Parameter(Misc.GenerateXavierQueryMatrix(inputSize, keySize, heads));
			exit = Misc.CreateXavierInitializedLinear(valueSize * heads, inputSize, true);
			this.normKernelSize = normKernelSize;
			this.epsilon = epsilon;
			keys = Misc.CreateXavierInitializedLinear(inputSize, keySize, false);
			values = Misc.CreateXavierInitializedLinear(inputSize, valueSize, false);
			RegisterComponents();
		}
	}







}
