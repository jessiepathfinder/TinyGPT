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
using System.Windows.Markup;
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
				using Tensor mean = input.sum(-1, true);
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
			inputs = Misc.CreateXavierInitializedLinear(size, size, true);
			output = Misc.CreateXavierInitializedLinear(size, size, false);
			gate = Parameter(ones(size));
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
					y1 = CustomActivations.ArLU(x);
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
	
	public sealed class MultiheadSelfAttention : Module<Tensor, Tensor>, IL2Regularizable
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
			bool doslice = slice > 0;
			long end = doslice ? input.size(0) : -1;
			using (NewDisposeScope())
			{
				Tensor x;
				using(Tensor key = MM3(input,keys)){
					using Tensor value = MM3(input, values);
					Tensor query;
					if(doslice){
						using Tensor y = input.slice(0, slice, end, 1);
						query = MM2(y,queries);
					} else{
						query = MM2(input, queries);
					}
					using(query)
					{
						x = Misc.MixedPrecisionAttention(query, key, value, mask, false, dropout);
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
					if(doslice ){
						using Tensor sliced = input.slice(0, slice, end, 1);
						x = y.addcmul(sliced, gate, one);
					} else{
						x = y.addcmul(input, gate, one);
					}
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
			Misc.L2RegularizeIMPL(queries, lambda);
			Misc.L2RegularizeIMPL(keys, lambda);
		}

		private readonly Linear exit;
		private readonly Parameter keys;
		private readonly Parameter queries;
		private readonly Parameter values;
		//private readonly Linear queries;
		//private readonly Parameter values;
		private readonly Parameter gate;
		private readonly int heads;
		public MultiheadSelfAttention(string name, int inputSize, int keySize, int heads, double epsilon, bool multiQuery) : base(name)
		{
			int effectiveHeads;
			if(multiQuery){
				effectiveHeads = 1;
				this.heads = heads;
			} else{
				effectiveHeads = heads;
			}
			keys = Parameter(Misc.GenerateKaimingQueryMatrix(inputSize, keySize, effectiveHeads));
			queries = Parameter(Misc.GenerateKaimingQueryMatrix(inputSize, keySize, heads));
			values = Parameter(Misc.GenerateKaimingQueryMatrix(inputSize, keySize, effectiveHeads));
			exit = Misc.CreateXavierInitializedLinear(keySize * heads, inputSize, false);
			gate = Parameter(ones(inputSize));
			this.epsilon = epsilon;
			bias = Parameter(zeros(inputSize));
			RegisterComponents();
		}

		private Tensor MM3(Tensor x, Tensor y){
			int mh = heads;
			Tensor c = MM2(x, y);
			if (mh == 0){
				return c;
			} else{
				Span<long> span = stackalloc long[4];
				span[0] = 1;
				span[1] = mh;
				span[2] = c.size(2);
				span[3] = c.size(3);
				using (c){
					return c.expand(span);
				}
			}
		}
		private static Tensor MM2(Tensor x, Tensor y){
			using Tensor z = x.matmul(y);
			return CustomActivations.HalfNorm(z);
		}
	}

	public sealed class TinyRNN : Module<Tensor, Tensor>
	{
		private readonly Linear input;
		private readonly Linear output;
		private readonly Linear core;
		private readonly Parameter gate;
		private readonly Parameter bias;
		private readonly Scalar epsilon;
		private static readonly Scalar one = 1.0;
		public TinyRNN(string name, int size, int coresize, double epsilon) : base(name)
		{
			this.epsilon = epsilon;
			int in2 = size + coresize;
			input = Misc.CreateManualKaimingInitializedLinear(size, coresize, true, in2, 1);
			output = Misc.CreateXavierInitializedLinear(coresize, size, false, 1);
			core = Misc.CreateManualKaimingInitializedLinear(coresize, coresize, false, in2, 1);
			gate = Parameter(ones(size));
			bias = Parameter(zeros(size));
			RegisterComponents();
		}

		public override Tensor forward(Tensor input1)
		{
			long len = input1.size(0);
			int len1 = (int)len;
			Tensor[] tensors = new Tensor[len1];
			Linear core = this.core;
			using (NewDisposeScope()){
				Tensor z1;
				using(NewDisposeScope()){
					using Tensor x = input.forward(input1);
					Tensor? state = null;
					for (int i = 0; i < len; ++i)
					{
						Tensor z = x.slice(0, i, i + 1, 1);
						if (state is { })
						{
							using Tensor y = z, y1 = core.forward(state);
							z = y.add(y1);
						}
						using (z)
						{
							state = z.arctan();
						}
						tensors[i] = state;
					}
					z1 = cat(tensors).MoveToOuterDisposeScope();
				}
				using (Tensor x = z1)
				{
					z1 = output.forward(x);
				}
				using (Tensor x = z1)
				{
					z1 = x.addcmul(input1, gate, one);
				}
				using (Tensor x = z1)
				{
					z1 = x.add(bias);
				}
				using (z1)
				{
					return CustomActivations.Norm(z1, epsilon).MoveToOuterDisposeScope();
				}
			}
		}
	}
	public sealed class SkipRNN : Module<Tensor, Tensor>
	{
		private readonly Linear input;
		private readonly Linear core;
		private readonly int stride;
		private readonly Linear output;
		private readonly Parameter gate;
		private readonly Parameter bias;
		private readonly Scalar epsilon;
		private static readonly Scalar one = 1.0;
		public SkipRNN(string name, int size, int coresize, int stride, double epsilon) : base(name)
		{
			int in2 = size + coresize;
			input = Misc.CreateManualKaimingInitializedLinear(size, coresize, true, in2, 1);
			output = Misc.CreateXavierInitializedLinear(coresize, size, false, 1);
			core = Misc.CreateManualKaimingInitializedLinear(coresize, coresize, false, in2, 1);
			gate = Parameter(ones(size));
			bias = Parameter(zeros(size));
			this.stride = stride;
			RegisterComponents();
			this.epsilon = epsilon;
		}

		public override Tensor forward(Tensor input1)
		{
			long len = input1.size(0);
			int len1 = (int)len;
			int stride = this.stride;
			int len2 = (len1 / stride) + Math.Min(len1 % stride, 1);
			Tensor[] tensors = new Tensor[len2];
			Linear core = this.core;
			using (NewDisposeScope())
			{
				Tensor z1;
				using (NewDisposeScope()){
					using Tensor x = input.forward(input1);
					Tensor? state = null;
					for (int i = 0; i < len2; ++i)
					{
						int i2 = i * stride;
						int len3 = Math.Min(i2 + stride, len1);
						Tensor z = x.slice(0, i2, len3, 1);
						len3 -= i2;
						if (state is { })
						{
							if (len3 < stride)
							{
								state = state.slice(0, 0, len3, 1);
							}
							using Tensor y = z, y1 = core.forward(state);
							z = y.add(y1);

						}
						using (z)
						{
							state = z.arctan();
						}
						tensors[i] = state;
					}
					z1 = cat(tensors).MoveToOuterDisposeScope();
				}
				using (Tensor x = z1)
				{
					z1 = output.forward(x);
				}
				using (Tensor x = z1)
				{
					z1 = x.addcmul(input1, gate, one);
				}
				using (Tensor x = z1)
				{
					z1 = x.add(bias);
				}
				using (z1)
				{
					return CustomActivations.Norm(z1, epsilon).MoveToOuterDisposeScope();
				}
			}
		}
	}

}

