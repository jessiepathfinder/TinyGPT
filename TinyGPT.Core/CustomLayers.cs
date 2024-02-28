using System;
using System.Buffers;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Serialization.Metadata;
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
		private static readonly Scalar one = 1;
		public static Tensor Tanh2(Tensor input){
			using(NewDisposeScope()){
				Tensor x;
				using(Tensor y = input.mul(input)){
					x = y.add(one);
				}
				return input.div(x).MoveToOuterDisposeScope();
			}
		}
		public static Tensor TanhELU(Tensor input)
		{
			using (NewDisposeScope())
			{
				using Tensor x = input.tanh();
				return x.max(input).MoveToOuterDisposeScope();
			}
		}
		public static Tensor ArLU(Tensor input)
		{
			using (NewDisposeScope())
			{
				using Tensor x = input.arctan();
				return x.max(input).MoveToOuterDisposeScope();
			}
		}
		public static Tensor KernelNorm(Tensor input, long kernelSize, Scalar epsilon){
			if(input.size(1) == kernelSize){
				return Norm(input, epsilon);
			}
			ScalarType scalarType = input.dtype;
			if (scalarType == ScalarType.Float64 | scalarType == ScalarType.Float32)
			{
				return KernelNormImpl(input, kernelSize, epsilon);	
			} else{
				using (NewDisposeScope())
				{
					Tensor y;
					using(Tensor x = input.to(ScalarType.Float32)){
						y = KernelNormImpl(x, kernelSize, epsilon);
					}
					using(y){
						return y.to(scalarType).MoveToOuterDisposeScope();
					}
				}
			}

		}
		public static Tensor Norm(Tensor input, Scalar epsilon)
		{
			ScalarType scalarType = input.dtype;
			if (scalarType == ScalarType.Float64 | scalarType == ScalarType.Float32)
			{
				return NormImpl(input, epsilon);
			}
			else
			{
				using (NewDisposeScope())
				{
					Tensor y;
					using (Tensor x = input.to(ScalarType.Float32))
					{
						y = NormImpl(x, epsilon);
					}
					using (y)
					{
						return y.to(scalarType).MoveToOuterDisposeScope();
					}
				}
			}

		}
		public static Tensor HalfNorm(Tensor input)
		{
			ScalarType scalarType = input.dtype;
			if (scalarType == ScalarType.Float64 | scalarType == ScalarType.Float32)
			{
				return HalfNormImpl(input);
			}
			else
			{
				using (NewDisposeScope())
				{
					Tensor y;
					using (Tensor x = input.to(ScalarType.Float32))
					{
						y = HalfNormImpl(x);
					}
					using (y)
					{
						return y.to(scalarType).MoveToOuterDisposeScope();
					}
				}
			}

		}
		private static Tensor KernelNormImpl(Tensor input, long kernelSize, Scalar epsilon)
		{
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
		private static Tensor NormImpl(Tensor input, Scalar epsilon){
			Tensor z;
			using (NewDisposeScope())
			{
				(Tensor std, Tensor mean) = input.std_mean(1, false, true);
				using (mean)
				{
					z = input.sub(mean);
				}
				using (Tensor x = std)
				{
					std = x.add(epsilon);
				}
				using (z){
					using (std)
					{
						return z.div(std).MoveToOuterDisposeScope();
					}
				}
				
			}
		}
		private static Tensor HalfNormImpl(Tensor input)
		{
			using (NewDisposeScope())
			{
				using Tensor mean = input.sum(1, true);
				return input.add(mean, (-1.0 / input.size(1))).MoveToOuterDisposeScope();
			}
		}
		public static Tensor LogReLU(Tensor input) {
			using(NewDisposeScope()){
				Tensor y;
				using(Tensor x = input.relu()){
					y = x.add(one);
				}
				using(y){
					return y.log().MoveToOuterDisposeScope();
				}
			}
		}
		public static Tensor CausalExponentalAverage(Tensor x, double decay)
		{
			int len = (int)x.size(0);
			Tensor[] tensors = new Tensor[len];
			Scalar decays = decay;
			Scalar growth = 1 - decay;

			using (NewDisposeScope())
			{
				Tensor? sum = null;



				for (int i = 0; i < len;)
				{
					int oi = i++;
					using Tensor current = x.slice(0, oi, i, 1);
					if (sum is null)
					{
						sum = current.mul(growth);
					}
					else
					{
						using (Tensor y = sum.mul(decays))
						{
							sum = y.add(current, growth);
						}
					}
					tensors[i] = sum;
				}
				return cat(tensors, 0).MoveToOuterDisposeScope();
			}
		}
	}
	public interface IL2Regularizable
	{
		public void L2Regularize(Scalar lambda);
	}

	public sealed class ResidualCausalConvolationalLookback : Module<Tensor, Tensor>
	{
		private readonly Linear input;
		private readonly Conv1d output;
		private readonly Parameter gate;
		private readonly Parameter bias;
		private readonly Scalar epsilon;
		private readonly int shift;
		
		private static readonly Scalar one = 1;
		public ResidualCausalConvolationalLookback(string name, int size, int compressedSize, int kernelSize, double epsilon) : base(name)
		{
			input = Misc.CreateKaimingInitializedLinear(size, compressedSize, false, init.FanInOut.FanIn);
			output = Conv1d(compressedSize, size, kernelSize, bias: false);
			gate = Parameter(ones(size));
			shift = (kernelSize) - 1;
			bias = Parameter(zeros(size));
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
				using (Tensor x = y)
				{
					y = output.forward(x);
				}


				using (Tensor x = y)
				{
					y = x.transpose(0, 1);
				}
				using (Tensor x = y)
				{
					y = x.addcmul(input1, gate, one);
				}
				using (Tensor x = y)
				{
					y = x.add(bias);
				}
				using (y)
				{
					return CustomActivations.Norm(y, epsilon).MoveToOuterDisposeScope();
				}
			}
		}
	}
	public sealed class ResidualComputeLayer : Module<Tensor, Tensor>
	{
		private readonly Linear inputs;
		private readonly Linear output;
		private readonly Parameter gate;
		private readonly Parameter arluBias;
		private static readonly Scalar one = 1;
		private readonly long arcore;
		private readonly Parameter bias;
		private readonly Scalar epsilon;
		public ResidualComputeLayer(string name, int size, double epsilon, long arluCoreUnits) : base(name)
		{
			if(arluCoreUnits < 1 | arluCoreUnits >= size){
				throw new ArgumentOutOfRangeException(nameof(arluCoreUnits));
			}
			arcore = arluCoreUnits;
			inputs = Misc.CreateXavierInitializedLinear(size, size, false);
			output = Misc.CreateXavierInitializedLinear(size, size, false);
			gate = Parameter(ones(size));
			arluBias = Parameter(ones(arluCoreUnits));
			bias = Parameter(zeros(size));
			this.epsilon = epsilon;
			RegisterComponents();
		}

		public override Tensor forward(Tensor input1)
		{
			using (NewDisposeScope())
			{

				Tensor y1;
				Tensor y2;
				using (Tensor x = inputs.forward(input1)){
					long my2 = arcore;
					y1 = x.slice(1, 0, my2, 1);
					y2 = x.slice(1, my2, input1.size(1), 1);
				}

				using (Tensor x = y1)
				{
					y1 = x.add(arluBias);
				}
				using (Tensor x = y1)
				{
					y1 = CustomActivations.ArLU(x);
				}


				using (Tensor x = y2)
				{
					y2 = CustomActivations.HalfNorm(x);
				}
				using (Tensor x = y2)
				{
					y2 = x.arctan();
				}

				Tensor y;
				using(y1){
					using(y2){
						y = cat(new Tensor[] { y1, y2 }, 1);
					}
				}
				using (Tensor x = y){
					y = output.forward(x);
				}


				using (Tensor x = y){
					y = x.addcmul(input1, gate, one);
				}
				using (Tensor x = y)
				{
					y = x.add(bias);
				}
				using (y)
				{
					return CustomActivations.Norm(y, epsilon).MoveToOuterDisposeScope();
				}
			}
		}
	}
	public sealed class LightweightMultiheadSelfAttention : Module<Tensor, Tensor>, IL2Regularizable
	{
		private static readonly Scalar one = 1;
		private readonly Parameter bias;
		private readonly Scalar epsilon;
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
					
					if (slice > 0)
					{
						long size = input.size(0);
						input = input.slice(0, slice, size, 1);
						Tensor a;
						Tensor b;
						using(Tensor d = z.slice(2, slice, size, 1)){
							a = d.slice(1, 0, 1, 1);
							b = d.slice(1, 1, heads, 1);
						}
						using (a){
							using(b){
								c = cat(new Tensor[] { b, a }, 1);
							}
						}
					} else{
						using Tensor a = z.slice(1, 0, 1, 1);
						using Tensor b = z.slice(1, 1, heads, 1);
						c = cat(new Tensor[] { b, a }, 1);
					}

					using (c)
					{
						x = Misc.MixedPrecisionAttention(c, z, z, mask, false, dropout);
					}
				}

				using (Tensor y = x)
				{
					x = y.squeeze(0);
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
					x = y.addcmul(input, gate, one);
				}
				using (Tensor y = x)
				{
					x = y.add(bias);
				}
				using (x)
				{
					return CustomActivations.Norm(x, epsilon).MoveToOuterDisposeScope();
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
		public LightweightMultiheadSelfAttention(string name, int inputSize, int keySize, int heads, double epsilon, double optimizerAssistedNormStrength) : base(name)
		{
			keys = Parameter(Misc.GenerateXavierQueryMatrix(inputSize, keySize, heads));
			exit = Misc.CreateXavierInitializedLinear(keySize * heads, inputSize, false);
			gate = Parameter(ones(inputSize));
			this.heads = heads;
			this.epsilon = epsilon;
			bias = Parameter(zeros(inputSize));
			//queries = Misc.CreateKaimingInitializedLinear(inputSize, keySize, false, init.FanInOut.FanIn);
			//values = Parameter(Misc.GenerateXavierQueryMatrix(inputSize, valueSize, heads));
			RegisterComponents();
		}
	}
	
	





}
