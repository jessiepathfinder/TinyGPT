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
using Module = TorchSharp.torch.nn.Module;

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
		public void L2Regularize(Scalar lambda);
	}
	public sealed class ResidualGatedCausalConvolationalLookback : Module<Tensor, Tensor>, IL2Regularizable
	{
		private readonly Linear input;
		private readonly Conv1d output;
		private readonly Conv1d gate;
		private readonly long normKernelSize;
		private readonly double epsilon;
		private readonly int shift;
		private static readonly Scalar one = 1;
		public ResidualGatedCausalConvolationalLookback(string name, int size, int compressedSize, int kernelSize, double epsilon, int normKernelSize) : base(name)
		{
			input = Misc.CreateKaimingInitializedLinear(size, compressedSize, false, init.FanInOut.FanIn);
			output = Conv1d(compressedSize, size, kernelSize);
			gate = Conv1d(compressedSize, size, kernelSize);
			shift = (kernelSize) - 1;
			
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
					y = x.transpose(0, 1);
				}
				using (Tensor x = y)
				{
					y = pad(x, (shift, 0), PaddingModes.Zeros, 0);
				}
				Tensor z;
				using (Tensor x = y)
				{
					y = output.forward(x);
					z = gate.forward(x);
				}
				using(Tensor x = z){
					z = x.sigmoid();
				}
				using(Tensor x = z, x2 = y){
					y = x2.mul(x);
					z = one - x;
				}
				using(Tensor x = z){
					z = x.transpose(1, 0);
				}
				using (Tensor x = y)
				{
					y = x.transpose(0, 1);
				}
				using (Tensor x = y)
				{
					Tensor z2;
					using(z){
						z2 = input1.mul(z);
					}
					using(z2){
						y = x.add(z2);
					}
				}
				using (y)
				{
					return CustomActivations.KernelNorm(y, normKernelSize, epsilon).MoveToOuterDisposeScope();
				}
			}
		}

		public void L2Regularize(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(input.weight, lambda);
			Misc.L2RegularizeIMPL(output.weight, lambda);
			Misc.L2RegularizeIMPL(gate.weight, lambda);
		}
	}
	public sealed class GatedResidualComputeLayer : Module<Tensor, Tensor>, IL2Regularizable
	{
		private readonly Linear input;
		private readonly Linear output;
		private readonly Linear gate;
		private readonly long normKernelSize;
		private readonly double epsilon;
		private static readonly Scalar one = 1;
		public GatedResidualComputeLayer(string name, int size, int coresize, double epsilon, int normKernelSize) : base(name)
		{
			input = Misc.CreateXavierInitializedLinear(size, coresize, true);
			output = Misc.CreateXavierInitializedLinear(coresize, size, true);
			gate = Misc.CreateXavierInitializedLinear(coresize, size, true);
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
				Tensor z;
				using(Tensor x = y){
					y = output.forward(x);
					z = gate.forward(x);
				}
				using (Tensor x = z, x2 = y)
				{
					y = x2.mul(x);
					z = one - x;
				}
				using (Tensor x = y){
					Tensor z2;
					using (z)
					{
						z2 = input1.mul(z);
					}
					using (z2)
					{
						y = x.add(z2);
					}
				}
				using (y)
				{
					return CustomActivations.KernelNorm(y, normKernelSize, epsilon).MoveToOuterDisposeScope();
				}
			}
		}

		public void L2Regularize(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(gate.weight, lambda);
			Misc.L2RegularizeIMPL(input.weight, lambda);
			Misc.L2RegularizeIMPL(output.weight, lambda);
		}
	}
	public sealed class LightweightMultiheadSelfAttention : Module<Tensor, Tensor>, IL2Regularizable
	{
		private readonly long normKernelSize;
		private readonly double epsilon;
		public override Tensor forward(Tensor input)
		{
			return Forward(input, 0, null);
		}
		public Tensor Forward(Tensor input, int slice, Tensor? mask = null, double dropout = 0.0)
		{
			using (NewDisposeScope())
			{
				Tensor x;
				using (Tensor z = input.matmul(keys)) {
					
					Tensor c;
					using (Tensor a = z.slice(0, 0, 1, 1))
					{
						using Tensor b = z.slice(0, 1, heads, 1);
						c = cat(new Tensor[] { b, a });
					}
					if (slice > 0)
					{
						long size = input.size(0);
						using (Tensor y = c){
							c = y.slice(1, slice, size, 1);
						}
						using (Tensor y = input)
						{
							input = y.slice(0, slice, size, 1);
						}
					}

					using (c)
					{
						x = Misc.MixedPrecisionAttention(c, z, z, mask, false, dropout);
					}
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
				using (Tensor y = x, z = input.mul(gate))
				{
					x = y.add(z);
				}
				using (x)
				{
					return CustomActivations.KernelNorm(x, normKernelSize, epsilon).MoveToOuterDisposeScope();
				}
			}


		}

		public void L2Regularize(Scalar lambda)
		{
			//Misc.L2RegularizeIMPL(queries.weight, lambda);
			Misc.L2RegularizeIMPL(keys, lambda);
		}

		private readonly Linear exit;
		private readonly Parameter keys;
		//private readonly Linear queries;
		//private readonly Parameter values;
		private readonly Parameter gate;
		private readonly int heads;

		public LightweightMultiheadSelfAttention(string name, int inputSize, int keySize, int heads, double epsilon, long normKernelSize) : base(name)
		{
			keys = Parameter(Misc.GenerateXavierQueryMatrix(inputSize, keySize, heads));
			exit = Misc.CreateXavierInitializedLinear(keySize * heads, inputSize, true);
			gate = Parameter(ones(inputSize));
			this.normKernelSize = normKernelSize;
			this.epsilon = epsilon;
			this.heads = heads;
			//queries = Misc.CreateKaimingInitializedLinear(inputSize, keySize, false, init.FanInOut.FanIn);
			//values = Parameter(Misc.GenerateXavierQueryMatrix(inputSize, valueSize, heads));
			RegisterComponents();
		}
	}
	public sealed class LongTermMemoryResidualComputeLayer : Module, IL2Regularizable
	{
		private readonly Linear inputGate;
		private readonly Linear input;
		private readonly Linear outputGate;
		private readonly Linear output;
		private static readonly Scalar one = 1;
		private static readonly double gain = 5.0 / 3.0;
		private readonly double epsilon;
		private readonly long normKernelSize;
		public LongTermMemoryResidualComputeLayer(string name, int size, double epsilon, long normKernelSize) : base(name)
		{
			inputGate = Misc.CreateXavierInitializedLinear(size, size, true);
			input = Misc.CreateXavierInitializedLinear(size, size, true);
			outputGate = Misc.CreateXavierInitializedLinear(size, size, true);
			output = Misc.CreateXavierInitializedLinear(size, size, true, gain);
			RegisterComponents();
			this.epsilon = epsilon;
			this.normKernelSize = normKernelSize;
		}
		public void Forward(ref Tensor hidden, ref Tensor? memory, bool disposemem){
			using(NewDisposeScope()){
				Tensor z;
				using (Tensor x = inputGate.forward(hidden))
				{
					z = x.sigmoid();
				}
				Tensor y;
				using (Tensor x = input.forward(hidden))
				{
					y = x.atan();
				}
				if (memory is null)
				{
					using (z)
					{
						using (y)
						{
							memory = y.mul(z);
						}
					}
				}
				else
				{
					using (Tensor x = y)
					{
						y = x.mul(z);
					}
					using (Tensor x = z)
					{
						z = one - x;
					}
					using (Tensor x = memory)
					{
						using (z)
						{
							memory = memory.mul(z);
						}
					}
					using (Tensor x = memory)
					{
						using (y)
						{
							memory = memory.add(y).MoveToOuterDisposeScope();
						}
					}
				}
				if (disposemem)
				{
					using(memory){
						z = outputGate.forward(hidden);
						y = output.forward(memory);
					}
					memory = null;
					using(Tensor x = z){
						z = x.sigmoid();
					}
					using(Tensor x = y){
						y = x.mul(z);
					}
				} else{
					using (Tensor x = outputGate.forward(hidden))
					{
						z = x.sigmoid();
					}
					using (Tensor x = output.forward(memory))
					{
						y = x.mul(z);
					}
				}
				

				using(Tensor x = z){
					z = one - x;
				}
				using (Tensor x = hidden){
					using(z){
						hidden = x.mul(z);
					}
				}
				using (Tensor x = hidden)
				{
					using (y)
					{
						hidden = x.add(y);
					}
				}
				using (Tensor x = hidden){
					hidden = CustomActivations.KernelNorm(x, normKernelSize, epsilon).MoveToOuterDisposeScope();
				}
				
			}
		}

		public void L2Regularize(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(input.weight, lambda);
			Misc.L2RegularizeIMPL(inputGate.weight, lambda);
			Misc.L2RegularizeIMPL(output.weight, lambda);
			Misc.L2RegularizeIMPL(outputGate.weight, lambda);
		}
	}
	public sealed class ResidualMultiQueryAttention : Module<Tensor, Tensor>, IL2Regularizable
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

		public void L2Regularize(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(exit.weight, lambda);
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
			keys = Misc.CreateKaimingInitializedLinear(inputSize, keySize, false, init.FanInOut.FanIn);
			values = Misc.CreateKaimingInitializedLinear(inputSize, valueSize, false, init.FanInOut.FanIn);
			RegisterComponents();
		}
	}







}
