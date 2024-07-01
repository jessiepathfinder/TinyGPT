using System;

using TorchSharp;
using TorchSharp.Modules;
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
	public sealed class DoesNothingModule : Module
	{
		private readonly Module inner;
		public DoesNothingModule(string name, Module inner) : base(name)
		{
			this.inner = inner;
			RegisterComponents();
		}
	}
	
	public static class CustomActivations
	{
		private static readonly Scalar one = 1;
		public static void ChannelDropout2(ref Tensor input, double prob){
			if(prob > 0.0){
				using Tensor x = input;
				using Tensor temp = ones(input.size(-1), input.dtype, input.device, false);
				input = x.mul(dropout(temp, prob, true, true));
			}
		}
		public static void Dropout(ref Tensor input, double prob)
		{
			if (is_grad_enabled() & prob > 0.0)
			{
				using Tensor x = input;
				input = dropout(x, prob, true, false);
			}
		}
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
		public static Tensor HalfNorm2(Tensor input, Scalar epsilon)
		{
			ScalarType scalarType = input.dtype;
			if (scalarType == ScalarType.Float64 | scalarType == ScalarType.Float32)
			{
				return HalfNorm2Impl(input, epsilon);
			}
			else
			{
				using (NewDisposeScope())
				{
					Tensor y;
					using (Tensor x = input.to(ScalarType.Float32))
					{
						y = HalfNorm2Impl(x, epsilon);
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
		private static Tensor HalfNorm2Impl(Tensor input, Scalar epsilon)
		{
			using (NewDisposeScope())
			{
				Tensor mean;
				using(Tensor x = input.mul(input)) mean = x.sum(-1, true);
				using (Tensor x = mean) mean = x.div(input.size(-1));
				using (Tensor x = mean) mean = x.sqrt();
				using (Tensor x = mean) mean = x.add(epsilon);
				using (mean){
					return input.div(mean).MoveToOuterDisposeScope();
				}
				
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


				CustomActivations.Dropout(ref y, dropout);

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
	public sealed class ResidualComputeLayer2 : Module<Tensor, Tensor>, IL2Regularizable, IL1Regularizable
	{
		private readonly Linear inputs;
		private readonly Linear output;
		//private readonly Parameter gate;
		private static readonly Scalar one = 1;
		private readonly Scalar epsilon;
		private readonly double dropout;
		public ResidualComputeLayer2(string name, int size, double epsilon, double init_output_gain, double dropout) : base(name)
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
					y = x.arctan();
				}


				CustomActivations.Dropout(ref y, dropout);

				using (Tensor x = y)
				{
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

	public sealed class MultiheadSelfAttention : Module<Tensor, Tensor>, IL2Regularizable, IL1Regularizable, ISelfAttention
	{
		private static readonly Scalar one = 1;
		private readonly Scalar epsilon;
		private readonly double aux_dropout;
		public override Tensor forward(Tensor input)
		{
			return Forward(input, 0, null);
		}
		private Tensor MM2(Tensor x, Tensor y){
			Tensor z = x.matmul(y);
			double axd = aux_dropout;
			if(axd > 0.0){
				using Tensor temp = ones(z.size(1), 1, z.size(3), x.dtype, x.device, false);
				dropout(temp, axd, true, true);
				using(z){
					return z.mul(temp);
				}
			}
			return z;
		}

		public Tensor Forward(Tensor input, int slice, Tensor? mask = null, double dropout = 0.0, bool causal = false)
		{
			bool doslice = slice > 0;
			long end = doslice ? input.size(0) : -1;
			
			using (NewDisposeScope())
			{
				Tensor x;
				using (Tensor key = input.matmul(keys)){
					using Tensor value = input.matmul(values);
					Tensor query;
					if(doslice){
						using Tensor y = input.slice(0, slice, end, 1);
						query = MM2(y, queries);
					} else{
						query = MM2(input, queries);
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
				CustomActivations.Dropout(ref x, aux_dropout);
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

		public void L1Regularize(Scalar lambda)
		{
			//Misc.L1RegularizeIMPL(values, lambda);
			//Misc.L1RegularizeIMPL(exit.weight, lambda);
		}

		private readonly Linear exit;
		private readonly Parameter keys;
		private readonly Parameter queries;
		private readonly Parameter values;
		//private readonly Linear queries;
		//private readonly Parameter values;
		//private readonly Parameter gate;
		public MultiheadSelfAttention(string name, int inputSize, int keySize, int heads, double epsilon, double init_gain, double keyQueryInitGain, double auxDropout) : base(name)
		{

			keys = Parameter(Misc.GenerateKaimingQueryMatrix(inputSize, keySize, heads, initial_gain: keyQueryInitGain));
			queries = Parameter(Misc.GenerateKaimingQueryMatrix(inputSize, keySize, heads, initial_gain: keyQueryInitGain));
			values = Parameter(Misc.GenerateKaimingQueryMatrix(inputSize, keySize, heads, initial_gain: init_gain));
			exit = Misc.CreateKaimingInitializedLinear(keySize * heads, inputSize, true, init.FanInOut.FanIn, init_gain);
			aux_dropout = auxDropout;
			//gate = Parameter(ones(inputSize));
			this.epsilon = epsilon;
			RegisterComponents();
		}
	}
	public interface ISelfAttention{
		public Tensor Forward(Tensor input, int slice, Tensor? mask = null, double dropout = 0.0, bool causal = false);
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

	//Reduced Parameter Count LSTM with residual bypass connections
	public sealed class KLSTM : Module<Tensor, Tensor>, IL1Regularizable, IL2Regularizable
	{
		private readonly Linear input;
		private readonly Linear output;
		private readonly Linear core;
		private readonly Scalar epsilon;
		
		private readonly Linear input_gate_;
		private readonly Linear input_gate;
		private readonly Linear output_gate_;
		private readonly Linear output_gate;
		private readonly double dropout;

		private static readonly Scalar one = 1.0;
		private static readonly Scalar two = 2.0;
		public KLSTM(string name, int size, int coresize, double epsilon, double dropout) : base(name)
		{
			this.epsilon = epsilon;

			int in2 = size + coresize;
			input = Misc.CreateManualKaimingInitializedLinear(size, coresize, true, in2, 1.0);
			core = Misc.CreateManualKaimingInitializedLinear(coresize, coresize, false, in2, 1.0);
			
			output_gate = Misc.CreateManualKaimingInitializedLinear(coresize, coresize, false, in2, 1.0);
			output_gate_ = Misc.CreateManualKaimingInitializedLinear(size, coresize, true, in2, 1.0);
			input_gate = Misc.CreateManualKaimingInitializedLinear(size, coresize, false, in2, 1.0);
			input_gate_ = Misc.CreateManualKaimingInitializedLinear(coresize, coresize, true, in2, 1.0);

			output = Misc.CreateKaimingInitializedLinear(coresize, size, true, init.FanInOut.FanIn, 1.0);


			RegisterComponents();
			this.dropout = dropout;

		}

		public override Tensor forward(Tensor input1)
		{
			return Forward(input1, 0);
		}
		public Tensor Forward(Tensor input1, int slice)
		{
			long len = input1.size(0);
			int len1 = (int)len;
			Tensor[] tensors = new Tensor[len1 - slice];
			Linear core = this.core;
			Linear reset = this.output_gate;
			Linear gate = this.input_gate;
			double dropout = this.dropout;
			Scalar one = KLSTM.one;
			Scalar two = KLSTM.two;
			using (NewDisposeScope())
			{
				Tensor z1;
				using (NewDisposeScope())
				{
					using Tensor x = input.forward(input1);
					using Tensor _gate = input_gate_.forward(input1);
					using Tensor _reset = output_gate_.forward(input1);
					Tensor? state = null;
					Tensor? memory = null;
					for (int i = 0; i < len; ++i)
					{
						Tensor arc = x.slice(0, i, i + 1, 1);
						if (state is null)
						{
							state = arc.arctan();
							Tensor myrst;
							using (Tensor y = _gate.slice(0, i, i + 1, 1)) myrst = y.negative();
							using (Tensor y = myrst) myrst = y.sigmoid();
							using (myrst)
							{
								using(state){
									memory = state.mul(myrst);
								}

							}
							using (Tensor y = _reset.slice(0, i, i + 1, 1)) myrst = y.neg();
							using (Tensor y = myrst) myrst = y.sigmoid();
							using(myrst){
								state = memory.mul(myrst);
							}

						}
						else
						{
							if(memory is null)throw new Exception("Null memory with non-null state (should not reach here)");
							Tensor myrst;
							using (Tensor y = arc, y2 = core.forward(state)) arc = y.add(y2);


							using (Tensor y = arc) arc = y.arctan();

							using (Tensor y = gate.forward(state), y1 = _gate.slice(0, i, i + 1, 1)) myrst = y.add(y1);
							using (Tensor y = myrst) myrst = y.sigmoid();
							using (Tensor y = memory) memory = y.mul(myrst);


							

							using (Tensor y = myrst) myrst = one - y;

							using(myrst){
								using(arc){
									using Tensor y = memory;
									memory = y.addcmul(arc, myrst, one);
								}
							}
							if(i > slice){
								using Tensor y = reset.forward(state), y1 = _reset.slice(0, i, i + 1, 1);
								myrst = y.add(y1);

							} else{
								using(state){
									myrst = reset.forward(state);
								}
								using Tensor y = myrst, y1 = _reset.slice(0, i, i + 1, 1);
								myrst = y.add(y1);
							}
							using (Tensor y = myrst) myrst = y.sigmoid();
							using (myrst)
							{
								state = myrst.mul(memory);
							}
							
						}
						CustomActivations.Dropout(ref state, dropout);
						using (Tensor y = state) state = y.mul(two);
						if (i < slice) continue;
						tensors[i - slice] = state;
					}
					z1 = cat(tensors).MoveToOuterDisposeScope();
				}

				using (Tensor x = z1)
				{
					z1 = output.forward(x);
				}
				using (Tensor x = z1)
				{
					if (slice == 0)
					{
						z1 = x.add(input1);
					}
					else
					{
						using Tensor z2 = input1.slice(0, slice, len, 1);
						z1 = x.add(z2);
					}
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
			Misc.L2RegularizeIMPL(output_gate.weight, lambda);
			Misc.L2RegularizeIMPL(input_gate_.weight, lambda);
			Misc.L2RegularizeIMPL(output_gate_.weight, lambda);
			Misc.L2RegularizeIMPL(input_gate.weight, lambda);
			
		}

		public void L1Regularize(Scalar lambda)
		{
			Misc.L1RegularizeIMPL(output.weight, lambda);
		}
	}
	public sealed class TinyMGU : Module<Tensor, Tensor>, IL1Regularizable, IL2Regularizable
	{
		private readonly Linear input;
		private readonly Linear output;
		private readonly Linear core;
		private readonly Scalar epsilon;
		private readonly Linear reset;
		private readonly Linear reset_;
		private readonly double dropout;

		private static readonly Scalar one = 1.0;
		public TinyMGU(string name, int size, int coresize, double epsilon, double dropout) : base(name)
		{
			this.epsilon = epsilon;


			input = Misc.CreateKaimingInitializedLinear(size, coresize, true, init.FanInOut.FanIn, 1.0);
			core = Misc.CreateZeroInitializedLinear(coresize, coresize, false);
			int in2 = size + coresize;
			reset_ = Misc.CreateManualKaimingInitializedLinear(size, coresize, true, in2, 1.0);
			reset = Misc.CreateManualKaimingInitializedLinear(coresize, coresize, false, in2, 1.0);

			output = Misc.CreateKaimingInitializedLinear(coresize, size, true, init.FanInOut.FanIn, 1.0);


			RegisterComponents();
			this.dropout = dropout;

		}

		public override Tensor forward(Tensor input1)
		{
			return Forward(input1, 0);
		}
		public Tensor Forward(Tensor input1, int slice)
		{
			long len = input1.size(0);
			int len1 = (int)len;
			Tensor[] tensors = new Tensor[len1 - slice];
			Linear core = this.core;
			Linear reset = this.reset;
			Scalar one = TinyMGU.one;
			using (NewDisposeScope())
			{
				Tensor z1;
				using (NewDisposeScope())
				{
					using Tensor x = input.forward(input1);
					using Tensor _reset = reset_.forward(input1);
					Tensor? state = null;
					for (int i = 0; i < len; ++i)
					{
						Tensor arc = x.slice(0, i, i + 1, 1);

						if (state is null)
						{
							using (arc)
							{
								state = arc.arctan();
							}
							Tensor myrst;
							using (Tensor y = _reset.slice(0, i, i + 1, 1)) myrst = y.negative();
							using (Tensor y = myrst) myrst = y.sigmoid();
							using (myrst)
							{
								using Tensor y = state;
								state = y.mul(myrst);
							}



						}
						else
						{
							Tensor myrst;
							using (Tensor y = reset.forward(state), y1 = _reset.slice(0, i, i + 1, 1)) myrst = y.add(y1);
							using (Tensor y = myrst) myrst = y.sigmoid();

							if(i > slice){
								state = myrst.mul(state);
							} else{
								using Tensor y = state;
								state = myrst.mul(y);
							}


							using (Tensor y = arc, y2 = core.forward(state)) arc = y.add(y2);



							using (Tensor y = myrst) myrst = one - y;
							using (Tensor y = arc) arc = y.arctan();
							using (myrst)
							{
								using (arc)
								{
									using Tensor y = state;
									state = y.addcmul(arc, myrst, one);
								}
							}
						}
						if (i < slice) continue;
						tensors[i - slice] = state;
					}
					z1 = cat(tensors).MoveToOuterDisposeScope();
				}
				CustomActivations.Dropout(ref z1, dropout);

				using (Tensor x = z1)
				{
					z1 = output.forward(x);
				}
				using (Tensor x = z1)
				{
					if(slice == 0){
						z1 = x.add(input1);
					} else{
						using Tensor z2 = input1.slice(0, slice, len, 1);
						z1 = x.add(z2);
					}
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
			Misc.L2RegularizeIMPL(reset.weight, lambda);
			Misc.L2RegularizeIMPL(reset_.weight, lambda);
		}

		public void L1Regularize(Scalar lambda)
		{
			Misc.L1RegularizeIMPL(output.weight, lambda);
		}
	}



	public sealed class Minitransformer : Module<Tensor, Tensor>, IL2Regularizable
	{
		private readonly Linear output;
		//private readonly Parameter gate;
		private static readonly Scalar one = 1;
		private readonly Scalar epsilon;
		private readonly int kernelSize;
		public Minitransformer(string name, int size, int kernel_size, double epsilon, double init_output_gain, double dropout) : base(name)
		{
			output = Misc.CreateXavierInitializedLinear(size, size, true, init_output_gain);
			//gate = Parameter(ones(size));
			this.epsilon = epsilon;
			RegisterComponents();
			kernelSize = kernel_size;
		}

		public override Tensor forward(Tensor input1)
		{
			int kern = kernelSize;
			using (NewDisposeScope())
			{
				Tensor y;	
				using (Tensor x = input1.transpose(0, 1))
				{
					y = avg_pool1d(x, kern, 1, kern - 1, true);
				}

				using (Tensor x = y) y = x.slice(1, 0, input1.size(0), 1);
				using (Tensor x = y) y = x.transpose(0, 1);


				using (Tensor x = y)
				{
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
			Misc.L1RegularizeIMPL(output.weight, lambda);
		}
	}

}

