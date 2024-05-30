using System;
using System.Buffers;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Serialization.Metadata;
using System.Threading.Tasks;
using System.Windows.Markup;
using TorchSharp;
using TorchSharp.Modules;
using static System.Runtime.InteropServices.JavaScript.JSType;
using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using Module = TorchSharp.torch.nn.Module;

namespace TinyGPT.Core
{
	public sealed class Word2VecStub : Module
	{
		public readonly Tensor hiddenWeights;
		public readonly Tensor wordEmbeddings;

		public Word2VecStub(string name, int latentTokenSize, int tokenClasses) : base(name)
		{
			hiddenWeights = zeros(latentTokenSize, latentTokenSize);
			wordEmbeddings = zeros(tokenClasses, latentTokenSize);
			RegisterComponents();
		}
	}
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
				return input.add(mean, (-1.0 / input.size(-1))).MoveToOuterDisposeScope();
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
	public interface IL1Regularizable
	{
		public void L1Regularize(Scalar lambda);
	}

	public sealed class ResidualCausalConvolationalLookback : Module<Tensor, Tensor>, IL2Regularizable
	{
		private readonly Linear input;
		private readonly Conv1d output;
		private readonly Scalar epsilon;
		private readonly int shift;
		
		private static readonly Scalar one = 1;
		public ResidualCausalConvolationalLookback(string name, int size, int compressedSize, int kernelSize, double epsilon, double init_gain) : base(name)
		{
			input = Misc.CreateKaimingInitializedLinear(size, compressedSize, false, init.FanInOut.FanIn, init_gain);
			output = Conv1d(compressedSize, size, kernelSize, bias: true);
			Tensor cvw = (output.weight ?? throw new Exception("Conv has no weights (should not reach here)"));
			using(no_grad()){
				cvw.normal_(0.0, init_gain / Math.Sqrt(compressedSize * kernelSize));
			}
			shift = (kernelSize) - 1;
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
					y = x.add(input1);
				}
				using (y)
				{
					return CustomActivations.Norm(y, epsilon).MoveToOuterDisposeScope();
				}
			}
		}

		public void L2Regularize(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(input.weight, lambda);
			Misc.L2RegularizeIMPL(output.weight, lambda);
		}
	}
	public sealed class ResidualComputeLayer : Module<Tensor, Tensor>, IL2Regularizable, IL1Regularizable
	{
		private readonly Linear inputs;
		private readonly Linear output;
		//private readonly Parameter gate;
		private static readonly Scalar one = 1;
		private readonly Scalar epsilon;
		private readonly double dropout;
		public ResidualComputeLayer(string name, int size, double epsilon, double init_output_gain, double dropout) : base(name)
		{
			inputs = Misc.CreateXavierInitializedLinear(size, size, true);
			output = Misc.CreateXavierInitializedLinear(size, size, true, init_output_gain);
			//gate = Parameter(ones(size));
			this.epsilon = epsilon;
			this.dropout = dropout;
			RegisterComponents();
		}

		public override Tensor forward(Tensor input1)
		{
			using (NewDisposeScope())
			{

				Tensor y;
				using (Tensor x = inputs.forward(input1))
				{
					y = x.sigmoid();
				}


				if(dropout > 0.0) {
					using Tensor x = y;
					y = functional.dropout(x, dropout);
				}
				
				using (Tensor x = y){
					y = output.forward(x);
				}
				using (Tensor x = y) y = x.add(input1); 
				
				using (y)
				{
					return CustomActivations.Norm(y, epsilon).MoveToOuterDisposeScope();
				}
			}
		}

		public void L2Regularize(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(inputs.weight, lambda);
		}

		public void L1Regularize(Scalar lambda)
		{
			Misc.L1RegularizeIMPL(output.weight, lambda);
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
		public Tensor Forward(Tensor input, int slice, Tensor? mask = null, double dropout = 0.0, bool causal = false)
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
						query = y.matmul(queries);
					} else{
						query = input.matmul(queries);
					}
					using(query)
					{
						x = Misc.MixedPrecisionAttention(query, key, value, mask, causal, dropout);
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
				/*
				using (Tensor y = x)
				{
					if(doslice ){
						using Tensor sliced = input.slice(0, slice, end, 1);
						x = y.addcmul(sliced, gate, one);
					} else{
						x = y.addcmul(input, gate, one);
					}
				}
				*/
				
				
				using (Tensor y = x)
				{
					if (doslice)
					{
						using Tensor sliced = input.slice(0, slice, end, 1);
						x = y.add(sliced);
					}
					else
					{
						x = y.add(input);
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
			Misc.L2RegularizeIMPL(values, lambda);
			Misc.L2RegularizeIMPL(exit.weight, lambda);
		}

		private readonly Linear exit;
		private readonly Parameter keys;
		private readonly Parameter queries;
		private readonly Parameter values;
		//private readonly Linear queries;
		//private readonly Parameter values;
		//private readonly Parameter gate;
		private readonly int heads;
		public MultiheadSelfAttention(string name, int inputSize, int keySize, int heads, double epsilon, bool multiQuery, double init_gain, double keyQueryInitGain) : base(name)
		{
			int effectiveHeads;
			if(multiQuery){
				effectiveHeads = 1;
				this.heads = heads;
			} else{
				effectiveHeads = heads;
			}
			keys = Parameter(Misc.GenerateKaimingQueryMatrix(inputSize, keySize, effectiveHeads, initial_gain: keyQueryInitGain));
			queries = Parameter(Misc.GenerateKaimingQueryMatrix(inputSize, keySize, heads, initial_gain: keyQueryInitGain));
			values = Parameter(Misc.GenerateKaimingQueryMatrix(inputSize, keySize, effectiveHeads, initial_gain: init_gain));
			exit = Misc.CreateKaimingInitializedLinear(keySize * heads, inputSize, false, init.FanInOut.FanIn,init_gain);
			//gate = Parameter(ones(inputSize));
			this.epsilon = epsilon;
			bias = Parameter(zeros(inputSize));
			RegisterComponents();
		}

		private Tensor MM3(Tensor x, Tensor y){
			int mh = heads;
			Tensor c = x.matmul(y);
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

	}
	public sealed class MultiValueSelfAttention : Module<Tensor, Tensor>, IL2Regularizable, IL1Regularizable
	{
		private static readonly Scalar one = 1;
		private readonly Parameter bias;
		private readonly Scalar epsilon;
		public override Tensor forward(Tensor input)
		{
			return Forward(input, 0, null);
		}
		public Tensor Forward(Tensor input, int slice, Tensor? mask = null, double dropout = 0.0, bool causal = false)
		{
			bool doslice = slice > 0;
			long end = doslice ? input.size(0) : -1;
			using (NewDisposeScope())
			{
				Tensor x;
				using (Tensor key = MM3(input, keys))
				{
					using Tensor value = input.matmul(values);
					Tensor query;
					if (doslice)
					{
						using Tensor y = input.slice(0, slice, end, 1);
						query = y.matmul(queries);
					}
					else
					{
						query = input.matmul(queries);
					}
					using (query)
					{
						x = Misc.MixedPrecisionAttention(query, key, value, mask, causal, dropout);
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
				/*
				using (Tensor y = x)
				{
					if(doslice ){
						using Tensor sliced = input.slice(0, slice, end, 1);
						x = y.addcmul(sliced, gate, one);
					} else{
						x = y.addcmul(input, gate, one);
					}
				}
				*/


				using (Tensor y = x)
				{
					if (doslice)
					{
						using Tensor sliced = input.slice(0, slice, end, 1);
						x = y.add(sliced);
					}
					else
					{
						x = y.add(input);
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
			Misc.L2RegularizeIMPL(values, lambda);
			
		}

		private readonly Linear exit;
		private readonly Parameter keys;
		private readonly Parameter queries;
		private readonly Parameter values;
		private readonly long heads;
		public MultiValueSelfAttention(string name, int inputSize, int keySize, int heads, double epsilon, double init_gain) : base(name)
		{

			keys = Parameter(Misc.GenerateKaimingQueryMatrix(inputSize, keySize, 1));
			queries = Parameter(Misc.GenerateKaimingQueryMatrix(inputSize, keySize, heads));
			values = Parameter(Misc.GenerateKaimingQueryMatrix(inputSize, keySize, heads, initial_gain: init_gain));
			exit = Misc.CreateKaimingInitializedLinear(keySize * heads, inputSize, false, init.FanInOut.FanIn, init_gain);
			//gate = Parameter(ones(inputSize));
			this.epsilon = epsilon;
			bias = Parameter(zeros(inputSize));
			RegisterComponents();
			this.heads = heads;
		}

		private Tensor MM3(Tensor x, Tensor y)
		{
			Tensor c = x.matmul(y);

			long heads = this.heads;
			if(heads == 1){
				return c;
			}

			Span<long> span = stackalloc long[4];
			span[0] = 1;
			span[1] = heads;
			span[2] = c.size(2);
			span[3] = c.size(3);
			using (c)
			{
				return c.expand(span);
			}
		}

		public void L1Regularize(Scalar lambda)
		{
			Misc.L1RegularizeIMPL(exit.weight, lambda);
		}
	}


	public sealed class GLTM : Module<Tensor, Tensor>
	{
		private readonly Linear output;

		private readonly Linear input_arctan;
		private readonly Linear input_gate;
		private readonly Linear forget_gate;

		private readonly Scalar epsilon;

		private static readonly Scalar one = 1.0;
		public GLTM(string name, int size, int longTermMemorySize, double epsilon) : base(name)
		{
			this.epsilon = epsilon;
			input_arctan = Misc.CreateKaimingInitializedLinear(size, longTermMemorySize, true, init.FanInOut.FanIn);
			input_gate = Misc.CreateKaimingInitializedLinear(size, longTermMemorySize, true, init.FanInOut.FanIn);
			forget_gate = Misc.CreateKaimingInitializedLinear(size, longTermMemorySize, true, init.FanInOut.FanIn);
			output = Misc.CreateKaimingInitializedLinear(longTermMemorySize, size, true, init.FanInOut.FanIn);


			RegisterComponents();
		}
		private static Tensor Srd(Tensor x){
			using(x){
				return x.sigmoid();
			}
		}
		private static Tensor Ard(Tensor x)
		{
			using (x)
			{
				return x.arctan();
			}
		}

		public override Tensor forward(Tensor input1)
		{
			int len = (int)input1.size(0);
			Tensor[] tensors = new Tensor[len];
			Tensor? state = null;
			using(NewDisposeScope()){


				Tensor x;
				using (NewDisposeScope()){
					using (Tensor data = Ard(input_arctan.forward(input1)), forget = Srd(forget_gate.forward(input1)), input = Srd(input_gate.forward(input1))){
						
						for (int i = 0; i < len; ++i)
						{
							Tensor mpled;
							using (Tensor mydata = data.slice(0, i, i + 1, 1), myinput = input.slice(0, i, i + 1, 1))
							{
								mpled = mydata * myinput;
							}
							if (state is null)
							{
								state = mpled;
							}
							else
							{
								using Tensor myforget = forget.slice(0, i, i + 1, 1);
								state = state.addcmul(mpled, myforget, one);
							}
							tensors[i] = state;
						}
					}
					x = cat(tensors, 0).MoveToOuterDisposeScope();
				}
				using(Tensor y = x)
					x = output.forward(y);
				using (Tensor y = x)
					x = y.add(input1);
				using(x){
					return CustomActivations.Norm(x, epsilon).MoveToOuterDisposeScope();
				}

			}

		}

		public void L2Regularize(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(input_arctan.weight, lambda);
			Misc.L2RegularizeIMPL(output.weight, lambda);
			Misc.L2RegularizeIMPL(forget_gate.weight, lambda);
			Misc.L2RegularizeIMPL(input_gate.weight, lambda);
		}
	}
	public sealed class TinyRNN : Module<Tensor, Tensor>, IL2Regularizable, IL1Regularizable
	{
		private readonly Linear input;
		private readonly Linear output;
		private readonly Linear core;
		private readonly Scalar epsilon;
		private static readonly Scalar one = 1.0;
		//private static readonly double sqrt2 = Math.Sqrt(2);
		public TinyRNN(string name, int size, int coresize, double epsilon) : base(name)
		{
			this.epsilon = epsilon;

			int in2 = size + coresize;
			input = Misc.CreateManualKaimingInitializedLinear(size, coresize, true, in2, 1);
			core = Misc.CreateManualKaimingInitializedLinear(coresize, coresize, false, in2, 1);
			output = Misc.CreateKaimingInitializedLinear(coresize, size, true, init.FanInOut.FanIn);
			

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
							state = z.sigmoid();
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
					z1 = x.add(input1);
				}
				using (z1)
				{
					return CustomActivations.Norm(z1, epsilon).MoveToOuterDisposeScope();
				}
			}
		}

		public void L2Regularize(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(input.weight, lambda);
		}

		public void L1Regularize(Scalar lambda)
		{
			Misc.L1RegularizeIMPL(output.weight, lambda);
			Misc.L1RegularizeIMPL(core.weight, lambda);
		}
	}
	public sealed class TinyGRU : Module<Tensor, Tensor>
	{
		private readonly Linear input;
		private readonly Linear output;
		private readonly Linear core;
		private readonly Scalar epsilon;
		private readonly Linear reset;
		private readonly Linear write;
		private readonly Linear reset_;
		private readonly Linear write_;

		private static readonly Scalar one = 1.0;
		public TinyGRU(string name, int size, int coresize, double epsilon) : base(name)
		{
			this.epsilon = epsilon;

			int in2 = size + coresize;
			input = Misc.CreateManualKaimingInitializedLinear(size, coresize, true, in2, 1);
			core = Misc.CreateManualKaimingInitializedLinear(coresize, coresize, false, in2, 1);
			reset_ = Misc.CreateManualKaimingInitializedLinear(size, coresize, true, in2, 1);
			reset = Misc.CreateManualKaimingInitializedLinear(coresize, coresize, false, in2, 1);
			write_ = Misc.CreateManualKaimingInitializedLinear(size, coresize, true, in2, 1);
			write = Misc.CreateManualKaimingInitializedLinear(coresize, coresize, false, in2, 1);

			output = Misc.CreateKaimingInitializedLinear(coresize, size, true, init.FanInOut.FanIn);


			RegisterComponents();
		}

		public override Tensor forward(Tensor input1)
		{
			long len = input1.size(0);
			int len1 = (int)len;
			Tensor[] tensors = new Tensor[len1];
			Linear core = this.core;
			using (NewDisposeScope())
			{
				Tensor z1;
				using (NewDisposeScope())
				{
					using Tensor x = input.forward(input1);
					using Tensor _reset = reset_.forward(input1);
					using Tensor _write = write_.forward(input1);
					Tensor? state = null;
					for (int i = 0; i < len; ++i)
					{
						Tensor arc = x.slice(0, i, i + 1, 1);
						
						Tensor w = _write.slice(0, i, i + 1, 1);
						if (state is { })
						{
							

							using(Tensor y = write.forward(arc), y1 = w){
								w = y.add(y1);
							}

							Tensor myrst;
							using (Tensor y = reset.forward(state), y1 = _reset.slice(0, i, i + 1, 1)) myrst = y.add(y1);
							using (Tensor y = myrst) myrst = y.sigmoid();

							using (Tensor y = myrst) myrst = y.mul(state);
							using (Tensor y = myrst) myrst = core.forward(y);
							using(myrst){
								using Tensor y = arc;
								arc = y.add(myrst);
							}

						}

						using (Tensor y = arc) arc = y.arctan();
						using (Tensor y = w) w = y.sigmoid();
						if(state is null){
							using(w){
								using(arc){
									state = w.mul(arc);
								}
							}
						} else{
							using (Tensor y = w)
							{
								using (Tensor y1 = arc) arc = y1.mul(y);
								w = one - y;
							}
							using (arc)
							{
								state = arc.addcmul(state, w, one);
							}
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
					z1 = x.add(input1);
				}
				
				using (z1)
				{
					return CustomActivations.Norm(z1, epsilon).MoveToOuterDisposeScope();
				}
			}
		}

		public void L2Regularize(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(core.weight, lambda);
			Misc.L2RegularizeIMPL(input.weight, lambda);
			Misc.L2RegularizeIMPL(output.weight, lambda);
			Misc.L2RegularizeIMPL(reset.weight, lambda);
			Misc.L2RegularizeIMPL(reset_.weight, lambda);
			Misc.L2RegularizeIMPL(write.weight, lambda);
			Misc.L2RegularizeIMPL(write_.weight, lambda);
		}
	}

}

