using System;
using System.Runtime.CompilerServices;
using Tensorboard;
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
	public interface IHaveSpecialTreatmentLayers{
		public IEnumerable<Parameter> GetSpecialTreatmentLayers();
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
		[ThreadStatic]public static bool threadPrivateDropoutDisableHack;
		public static void ChannelDropout2(ref Tensor input, double prob){
			if((prob > 0.0 & !threadPrivateDropoutDisableHack) && is_grad_enabled() && input.requires_grad)
			{
				using Tensor x = input;
				using Tensor temp = ones(input.size(-1), input.dtype, input.device, false);
				input = x.mul(dropout(temp, prob, true, true));
			}
		}
		public static void Dropout(ref Tensor input, double prob)
		{
			if ((prob > 0.0 & !threadPrivateDropoutDisableHack) && is_grad_enabled() && input.requires_grad)
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

		private static jit.CompilationUnit BF16_layerNormKernel = jit.compile("def mixed_layer_norm(inp : Tensor, epsilon : float) -> Tensor:\r\n    inp = inp.sub(inp.mean(-1,keepdim=True))\r\n    return inp.div(inp.mul(inp).mean(-1,keepdim=True).sqrt().add(epsilon))\r\n");

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
			return BF16_layerNormKernel.invoke<Tensor>("mixed_layer_norm", input, epsilon);
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
	public interface IL2Regularizable2
	{
		public void L2RegularizeOutput(Scalar lambda);
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
	public sealed class ResidualComputeLayer2 : Module<Tensor, Tensor>, IL2Regularizable, IL1Regularizable, IL2Regularizable2
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
		public void L2RegularizeOutput(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(output.weight, lambda);
		}

		public void L1Regularize(Scalar lambda)
		{
			Misc.L1RegularizeIMPL(output.weight, lambda);
		}
	}
	public sealed class ResidualComputeLayer3 : Module<Tensor, Tensor>, IL2Regularizable
	{
		private readonly Linear inputs;
		private readonly Linear output;
		//private readonly Parameter gate;
		private readonly Scalar epsilon;




		public ResidualComputeLayer3(string name, int size, double epsilon, double init_output_gain) : base(name)
		{
			inputs = Misc.CreateXavierInitializedLinear(size, size, true);
			output = Misc.CreateXavierInitializedLinear(size, size, true, init_output_gain);
			//gate = Parameter(ones(size));
			this.epsilon = epsilon;
			RegisterComponents();
		}

		public override Tensor forward(Tensor input1)
		{
			using (NewDisposeScope())
			{

				Tensor y;
				using (Tensor x = inputs.forward(input1))
				{
					y = x.softplus();
				}



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
			Misc.L2RegularizeIMPL(output.weight, lambda);
		}
	}

	public sealed class MultiheadSelfAttention : Module<Tensor, Tensor>, IL2Regularizable, ISelfAttention, IHaveSpecialTreatmentLayers
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
			if((axd > 0.0 & !CustomActivations.threadPrivateDropoutDisableHack) && is_grad_enabled()){
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

		public IEnumerable<Parameter> GetSpecialTreatmentLayers()
		{
			return new Parameter[] { keys, queries };
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
	public sealed class NormalizedMultiheadCausalSelfAttention : Module<Tensor, Tensor>
	{
		private static readonly Scalar one = 1;
		private readonly Scalar epsilon;
		private readonly double aux_dropout;

		private Tensor MM2(Tensor x, Tensor y)
		{
			Tensor z = x.matmul(y);
			double axd = aux_dropout;
			if ((axd > 0.0 & !CustomActivations.threadPrivateDropoutDisableHack) && is_grad_enabled())
			{
				using Tensor temp = ones(z.size(1), 1, z.size(3), x.dtype, x.device, false);
				dropout(temp, axd, true, true);
				using (z)
				{
					return z.mul(temp);
				}
			}
			return z;
		}

		public override Tensor forward(Tensor input)
		{

			using (NewDisposeScope())
			{
				Tensor x;
				using (Tensor key = input.matmul(keys))
				{
					using Tensor value = input.matmul(values);
					using Tensor query = MM2(input,queries);
					
					using (query)
					{
						x = Misc.MixedPrecisionAttention(query, key, value, null, true);
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
				}using (Tensor y = x)
				{
					x = y.add(input);
				}

				


				using (Tensor y = x)
				{
					x = compilationUnit.invoke<Tensor>("aot_history_norm", y, mynorm);
				}
				using (x)
				{
					return CustomActivations.Norm(x, epsilon).MoveToOuterDisposeScope();
				}
			}


		}
		private static jit.CompilationUnit compilationUnit = jit.compile("def aot_history_norm(_input : Tensor, decay : float):\r\n    length = _input.size(-2)\r\n    bias = _input.select(-2,0)\r\n    corrections = [bias.unsqueeze(-2)]\r\n    for x in range(1,length):\r\n        bias = _input.select(-2,x).add(bias,alpha=decay)\r\n        corrections.append(bias.unsqueeze(-2))\r\n    return _input.add(torch.cat(corrections, -2), alpha = decay - 1.0)\r\n");


		public IEnumerable<Parameter> GetSpecialTreatmentLayers()
		{
			return new Parameter[] { keys, queries };
		}

		private readonly Linear exit;
		private readonly Parameter keys;
		private readonly Parameter queries;
		private readonly Parameter values;
		private readonly Scalar mynorm;
		//private readonly Linear queries;
		//private readonly Parameter values;
		//private readonly Parameter gate;
		public NormalizedMultiheadCausalSelfAttention(string name, int inputSize, int keySize, int heads, double epsilon, double init_gain, double keyQueryInitGain, double auxDropout, double normalization) : base(name)
		{

			keys = Parameter(Misc.GenerateKaimingQueryMatrix(inputSize, keySize, heads, initial_gain: keyQueryInitGain));
			queries = Parameter(Misc.GenerateKaimingQueryMatrix(inputSize, keySize, heads, initial_gain: keyQueryInitGain));
			values = Parameter(Misc.GenerateKaimingQueryMatrix(inputSize, keySize, heads, initial_gain: init_gain));
			exit = Misc.CreateKaimingInitializedLinear(keySize * heads, inputSize, false, init.FanInOut.FanIn, init_gain);
			aux_dropout = auxDropout;
			mynorm = normalization;
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

	//Reduced Parameter Count LSTM with residual bypass connection
	[Obsolete("Broken input gate calculation", false)]
	public sealed class AOT_KLSTM : Module<Tensor, Tensor>, IL1Regularizable, IL2Regularizable, IL2Regularizable2
	{
		private readonly Parameter input;
		
		private readonly Parameter core;
		private readonly Parameter core_bias;
		private readonly Scalar epsilon;

		private readonly Parameter input_gate_;
		private readonly Parameter input_gate;
		private readonly Parameter input_gate_bias;

		private readonly Parameter output_gate_;
		private readonly Parameter output_gate;
		private readonly Parameter output_gate_bias;

		private readonly Parameter output;
		private readonly Parameter output_bias;

		private readonly double dropout;

		

		private static readonly Scalar two = 2.0;
		private readonly int coresize;
		public AOT_KLSTM(string name, int size, int coresize, double epsilon, double dropout) : base(name)
		{
			this.epsilon = epsilon;

			Scalar std = Math.Sqrt(size + coresize);
			using(no_grad()){
				input = Parameter(randn(size, coresize).div_(std));
				input_gate_ = Parameter(randn(size, coresize).div_(std));
				output_gate_ = Parameter(randn(size, coresize).div_(std));

				core = Parameter(randn(coresize, coresize).div_(std));
				input_gate = Parameter(randn(coresize, coresize).div_(std));
				output_gate = Parameter(randn(coresize, coresize).div_(std));

				core_bias = Parameter(zeros(coresize));
				input_gate_bias = Parameter(zeros(coresize));
				output_gate_bias = Parameter(zeros(coresize));
				output_bias = Parameter(zeros(size));

				output = Parameter(randn(coresize, size).div_(Math.Sqrt(coresize)));
			}



			RegisterComponents();
			this.dropout = dropout;
			this.coresize = coresize;
		}
		[ThreadStatic]
		public static bool forceDropoutNonGrad;
		public override Tensor forward(Tensor input1)
		{
			double dropout = this.dropout;
			Tensor x;
			if((dropout > 0.0 & !CustomActivations.threadPrivateDropoutDisableHack) && (forceDropoutNonGrad || (is_grad_enabled() && input1.requires_grad))){
				using Tensor dropoutMask = empty(input1.size(0), input.size(1), input1.dtype, input1.device);
				dropoutMask.fill_(two);

				functional.dropout(dropoutMask, dropout, true, true);

				//TorchSharp bug: CANNOT use dropout kernel in TorchScript
				//Jessie Lesbian High-Intelligence Solution: AOT dropout mask
				x = compilationUnit.invoke<Tensor>("aot_klstm_core", input_gate, output_gate, core, input_gate_, input_gate_bias, output_gate_, output_gate_bias, input, core_bias, output, output_bias, input1, dropoutMask);
			} else{
				x = compilationUnit.invoke<Tensor>("aot_klstm_core_nodrop", input_gate, output_gate, core, input_gate_, input_gate_bias, output_gate_, output_gate_bias, input, core_bias, output, output_bias, input1);
			}
			using(x){
				return CustomActivations.Norm(x, epsilon);
			}
		}
		private static readonly jit.CompilationUnit compilationUnit = jit.compile("def aot_klstm_core(input_gate : Tensor,output_gate : Tensor,core : Tensor, input_gate_ : Tensor, input_gate_bias : Tensor, output_gate_ : Tensor, output_gate_bias : Tensor, input_ : Tensor, input_bias : Tensor, output_ : Tensor, output_bias_ : Tensor, _input : Tensor, dropoutMask : Tensor) -> Tensor:\r\n    \r\n    length = _input.size(-2)\r\n    \r\n    input_gate_ = _input.matmul(input_gate_).add(input_gate_bias)\r\n    \r\n    output_gate_ = _input.matmul(output_gate_).add(output_gate_bias)\r\n    input_ = _input.matmul(input_).add(input_bias)\r\n    \r\n    \r\n    cell_state = input_.select(-2,0).arctan().mul(input_gate.select(-2,0).negative().sigmoid())\r\n    \r\n    hidden_state = cell_state.mul(output_gate.select(-2,0).sigmoid()).mul(dropoutMask.select(-2,0))\r\n    if(length == 1):\r\n        return hidden_state.unsqueeze(-2).matmul(output_).add(output_bias_).add(_input)\r\n    \r\n    outputs = [hidden_state.unsqueeze(-2)]\r\n    \r\n    for x in range(1,length):\r\n        gate = hidden_state.matmul(input_gate).add(input_gate_.select(-2,x)).sigmoid()\r\n        cell_state = cell_state.mul(gate).addcmul(input_.select(-2,x).add(hidden_state.matmul(core)).arctan(), gate.sub(1.0).negative())\r\n        hidden_state = cell_state.mul(output_gate_.select(-2,x).add(hidden_state.matmul(output_gate)).sigmoid()).mul(dropoutMask.select(-2,x))\r\n        outputs.append(hidden_state.unsqueeze(-2))\r\n        pass\r\n    return torch.cat(outputs, -2).matmul(output_).add(output_bias_).add(_input)\r\n    \r\ndef aot_klstm_core_nodrop(input_gate : Tensor,output_gate : Tensor,core : Tensor, input_gate_ : Tensor, input_gate_bias : Tensor, output_gate_ : Tensor, output_gate_bias : Tensor, input_ : Tensor, input_bias : Tensor, output_ : Tensor, output_bias_ : Tensor, _input : Tensor) -> Tensor:\r\n    \r\n    length = _input.size(-2)\r\n    \r\n    input_gate_ = _input.matmul(input_gate_).add(input_gate_bias)\r\n    \r\n    output_gate_ = _input.matmul(output_gate_).add(output_gate_bias)\r\n    input_ = _input.matmul(input_).add(input_bias)\r\n    \r\n    \r\n    cell_state = input_.select(-2,0).arctan().mul(input_gate.select(-2,0).negative().sigmoid())\r\n    \r\n    hidden_state = cell_state.mul(output_gate.select(-2,0).sigmoid()).mul(2.0)\r\n    if(length == 1):\r\n        return hidden_state.unsqueeze(-2).matmul(output_).add(output_bias_).add(_input)\r\n    \r\n    outputs = [hidden_state.unsqueeze(-2)]\r\n    \r\n    for x in range(1,length):\r\n        gate = hidden_state.matmul(input_gate).add(input_gate_.select(-2,x)).sigmoid()\r\n        cell_state = cell_state.mul(gate).addcmul(input_.select(-2,x).add(hidden_state.matmul(core)).arctan(), gate.sub(1.0).negative())\r\n        hidden_state = cell_state.mul(output_gate_.select(-2,x).add(hidden_state.matmul(output_gate)).sigmoid()).mul(2.0)\r\n        outputs.append(hidden_state.unsqueeze(-2))\r\n        pass\r\n    return torch.cat(outputs, -2).matmul(output_).add(output_bias_).add(_input)\r\n");
		
		public void L2Regularize(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(input, lambda);
			Misc.L2RegularizeIMPL(output_gate, lambda);
			Misc.L2RegularizeIMPL(input_gate_, lambda);
			Misc.L2RegularizeIMPL(output_gate_, lambda);
			Misc.L2RegularizeIMPL(input_gate, lambda);

		}

		public void L1Regularize(Scalar lambda)
		{
			Misc.L1RegularizeIMPL(output, lambda);
		}

		public void L2RegularizeOutput(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(output, lambda);
		}

	}

	public sealed class AOT_KLSTM_Bugfix : Module<Tensor, Tensor>, IL1Regularizable, IL2Regularizable, IL2Regularizable2
	{
		private readonly Parameter input;

		private readonly Parameter core;
		private readonly Parameter core_bias;
		private readonly Scalar epsilon;

		private readonly Parameter input_gate_;
		private readonly Parameter input_gate;
		private readonly Parameter input_gate_bias;

		private readonly Parameter output_gate_;
		private readonly Parameter output_gate;
		private readonly Parameter output_gate_bias;

		private readonly Parameter output;
		private readonly Parameter output_bias;

		private readonly double dropout;



		private static readonly Scalar two = 2.0;
		private readonly int coresize;
		public AOT_KLSTM_Bugfix(string name, int size, int coresize, double epsilon, double dropout) : base(name)
		{
			this.epsilon = epsilon;

			Scalar std = Math.Sqrt(size + coresize);
			using (no_grad())
			{
				input = Parameter(randn(size, coresize).div_(std));
				input_gate_ = Parameter(randn(size, coresize).div_(std));
				output_gate_ = Parameter(randn(size, coresize).div_(std));

				core = Parameter(randn(coresize, coresize).div_(std));
				input_gate = Parameter(randn(coresize, coresize).div_(std));
				output_gate = Parameter(randn(coresize, coresize).div_(std));

				core_bias = Parameter(zeros(coresize));
				input_gate_bias = Parameter(zeros(coresize));
				output_gate_bias = Parameter(zeros(coresize));
				output_bias = Parameter(zeros(size));

				output = Parameter(randn(coresize, size).div_(Math.Sqrt(coresize)));
			}



			RegisterComponents();
			this.dropout = dropout;
			this.coresize = coresize;
		}
		[ThreadStatic]
		public static bool forceDropoutNonGrad;
		public override Tensor forward(Tensor input1)
		{
			double dropout = this.dropout;
			Tensor x;
			if ((dropout > 0.0 & !CustomActivations.threadPrivateDropoutDisableHack) && (forceDropoutNonGrad || (is_grad_enabled() && input1.requires_grad)))
			{
				using Tensor dropoutMask = empty(input1.size(0), input.size(1), input1.dtype, input1.device);
				dropoutMask.fill_(two);

				functional.dropout(dropoutMask, dropout, true, true);

				//TorchSharp bug: CANNOT use dropout kernel in TorchScript
				//Jessie Lesbian High-Intelligence Solution: AOT dropout mask
				x = compilationUnit.invoke<Tensor>("aot_klstm_core", input_gate, output_gate, core, input_gate_, input_gate_bias, output_gate_, output_gate_bias, input, core_bias, output, output_bias, input1, dropoutMask);
			}
			else
			{
				x = compilationUnit.invoke<Tensor>("aot_klstm_core_nodrop", input_gate, output_gate, core, input_gate_, input_gate_bias, output_gate_, output_gate_bias, input, core_bias, output, output_bias, input1);
			}
			using (x)
			{
				return CustomActivations.Norm(x, epsilon);
			}
		}
		private static readonly jit.CompilationUnit compilationUnit = jit.compile("def aot_klstm_core(input_gate : Tensor,output_gate : Tensor,core : Tensor, input_gate_ : Tensor, input_gate_bias : Tensor, output_gate_ : Tensor, output_gate_bias : Tensor, input_ : Tensor, input_bias : Tensor, output_ : Tensor, output_bias_ : Tensor, _input : Tensor, dropoutMask : Tensor) -> Tensor:\r\n    \r\n    length = _input.size(-2)\r\n    \r\n    input_gate_ = _input.matmul(input_gate_).add(input_gate_bias)\r\n    \r\n    output_gate_ = _input.matmul(output_gate_).add(output_gate_bias)\r\n    input_ = _input.matmul(input_).add(input_bias)\r\n    \r\n    \r\n    cell_state = input_.select(-2,0).arctan().mul(input_gate_.select(-2,0).negative().sigmoid())\r\n    \r\n    hidden_state = cell_state.mul(output_gate_.select(-2,0).sigmoid()).mul(dropoutMask.select(-2,0))\r\n    if(length == 1):\r\n        return hidden_state.unsqueeze(-2).matmul(output_).add(output_bias_).add(_input)\r\n    \r\n    outputs = [hidden_state.unsqueeze(-2)]\r\n    \r\n    for x in range(1,length):\r\n        gate = hidden_state.matmul(input_gate).add(input_gate_.select(-2,x)).sigmoid()\r\n        cell_state = cell_state.mul(gate).addcmul(input_.select(-2,x).add(hidden_state.matmul(core)).arctan(), gate.sub(1.0).negative())\r\n        hidden_state = cell_state.mul(output_gate_.select(-2,x).add(hidden_state.matmul(output_gate)).sigmoid()).mul(dropoutMask.select(-2,x))\r\n        outputs.append(hidden_state.unsqueeze(-2))\r\n        pass\r\n    return torch.cat(outputs, -2).matmul(output_).add(output_bias_).add(_input)\r\n    \r\ndef aot_klstm_core_nodrop(input_gate : Tensor,output_gate : Tensor,core : Tensor, input_gate_ : Tensor, input_gate_bias : Tensor, output_gate_ : Tensor, output_gate_bias : Tensor, input_ : Tensor, input_bias : Tensor, output_ : Tensor, output_bias_ : Tensor, _input : Tensor) -> Tensor:\r\n    \r\n    length = _input.size(-2)\r\n    \r\n    input_gate_ = _input.matmul(input_gate_).add(input_gate_bias)\r\n    \r\n    output_gate_ = _input.matmul(output_gate_).add(output_gate_bias)\r\n    input_ = _input.matmul(input_).add(input_bias)\r\n    \r\n    \r\n    cell_state = input_.select(-2,0).arctan().mul(input_gate_.select(-2,0).negative().sigmoid())\r\n    \r\n    hidden_state = cell_state.mul(output_gate_.select(-2,0).sigmoid()).mul(2.0)\r\n    if(length == 1):\r\n        return hidden_state.unsqueeze(-2).matmul(output_).add(output_bias_).add(_input)\r\n    \r\n    outputs = [hidden_state.unsqueeze(-2)]\r\n    \r\n    for x in range(1,length):\r\n        gate = input_gate_.select(-2,x).add(hidden_state.matmul(input_gate)).sigmoid()\r\n        cell_state = cell_state.mul(gate).addcmul(input_.select(-2,x).add(hidden_state.matmul(core)).arctan(), gate.sub(1.0).negative())\r\n        hidden_state = cell_state.mul(output_gate_.select(-2,x).add(hidden_state.matmul(output_gate)).sigmoid()).mul(2.0)\r\n        outputs.append(hidden_state.unsqueeze(-2))\r\n        pass\r\n    return torch.cat(outputs, -2).matmul(output_).add(output_bias_).add(_input)\r\n");

		public void L2Regularize(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(input, lambda);
			Misc.L2RegularizeIMPL(core, lambda);
			Misc.L2RegularizeIMPL(output_gate, lambda);
			Misc.L2RegularizeIMPL(input_gate_, lambda);
			Misc.L2RegularizeIMPL(output_gate_, lambda);
			Misc.L2RegularizeIMPL(input_gate, lambda);

		}

		public void L1Regularize(Scalar lambda)
		{
			Misc.L1RegularizeIMPL(output, lambda);
		}

		public void L2RegularizeOutput(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(output, lambda);
		}
	}
	public sealed class AOT_KLSTM_Peephole : Module<Tensor, Tensor>, IL1Regularizable, IL2Regularizable, IL2Regularizable2
	{
		private readonly Parameter input;

		private readonly Parameter core;
		private readonly Parameter core_bias;
		private readonly Scalar epsilon;

		private readonly Parameter reset_gate_;
		private readonly Parameter reset_gate;
		private readonly Parameter reset_gate_bias;

		private readonly Parameter input_gate_;
		private readonly Parameter input_gate;
		private readonly Parameter input_gate_bias;

		private readonly Parameter output_gate_;
		private readonly Parameter output_gate;
		private readonly Parameter output_gate_bias;

		private readonly Parameter output;
		private readonly Parameter output_bias;




		private static readonly Scalar two = 2.0;
		public AOT_KLSTM_Peephole(string name, int size, int coresize, double epsilon) : base(name)
		{
			this.epsilon = epsilon;

			Scalar std = Math.Sqrt(size + coresize);
			using (no_grad())
			{
				input = Parameter(randn(size, coresize).div_(std));
				input_gate_ = Parameter(randn(size, coresize).div_(std));
				input_gate_bias = Parameter(zeros(coresize));

				reset_gate = Parameter(randn(coresize, coresize).div_(std));
				reset_gate_ = Parameter(randn(size, coresize).div_(std));
				reset_gate_bias = Parameter(zeros(coresize));

				output_gate_ = Parameter(randn(size, coresize).div_(std));

				core = Parameter(randn(coresize, coresize).div_(std));
				input_gate = Parameter(randn(coresize, coresize).div_(std));
				output_gate = Parameter(randn(coresize, coresize).div_(std));

				core_bias = Parameter(zeros(coresize));

				output_gate_bias = Parameter(zeros(coresize));
				output_bias = Parameter(zeros(size));

				output = Parameter(randn(coresize * 2, size).div_(Math.Sqrt(size)));
			}



			RegisterComponents();
		}

		public override Tensor forward(Tensor input1)
		{
			if (input1.size(-2) < 2) throw new Exception("Unit or zero length input not accepted!");
			using (Tensor x = compilationUnit.invoke<Tensor>("aot_klstm_core_nodrop_peephole", input_gate, reset_gate, output_gate, core, input_gate_, input_gate_bias, reset_gate_, reset_gate_bias, output_gate_, output_gate_bias, input, core_bias, output, output_bias, input1))
			{
				return CustomActivations.Norm(x, epsilon);
			}
		}
		private static readonly jit.CompilationUnit compilationUnit = jit.compile("def aot_klstm_core_nodrop_peephole(input_gate : Tensor, reset_gate : Tensor,output_gate : Tensor,core : Tensor, input_gate_ : Tensor, input_gate_bias : Tensor, reset_gate_ : Tensor, reset_gate_bias : Tensor, output_gate_ : Tensor, output_gate_bias : Tensor, input_ : Tensor, input_bias : Tensor, output_ : Tensor, output_bias_ : Tensor, _input : Tensor) -> Tensor:\r\n    \r\n    length = _input.size(-2)\r\n    \r\n    input_gate_ = _input.matmul(input_gate_).add(input_gate_bias)\r\n    reset_gate_ = _input.transpose(-2,0)[1:length].transpose(-2,0).matmul(reset_gate_).add(reset_gate_bias)\r\n    \r\n    output_gate_ = _input.matmul(output_gate_).add(output_gate_bias)\r\n    input_ = _input.matmul(input_).add(input_bias)\r\n    \r\n    \r\n    cell_state = input_.select(-2,0).arctan().mul(input_gate_.select(-2,0).negative().sigmoid())\r\n    \r\n    hidden_state = cell_state.mul(output_gate_.select(-2,0).sigmoid()).mul(2.0)\r\n\r\n    \r\n    outputs = [torch.cat([hidden_state.unsqueeze(-2), cell_state.unsqueeze(-2)], -1)]\r\n    \r\n    for x in range(1,length):\r\n        gate = cell_state.matmul(input_gate).add(input_gate_.select(-2,x)).sigmoid()\r\n        cell_state = cell_state.mul(gate).addcmul(input_.select(-2,x).add(cell_state.mul(reset_gate_.select(-2, x - 1).add(cell_state.matmul(reset_gate)).sigmoid()).matmul(core),alpha=2.0).arctan(), gate.sub(1.0).negative())\r\n        hidden_state = cell_state.mul(output_gate_.select(-2,x).add(hidden_state.matmul(output_gate)).sigmoid()).mul(2.0)\r\n        outputs.append(torch.cat([hidden_state.unsqueeze(-2),cell_state.unsqueeze(-2)],-1))\r\n        pass\r\n    return torch.cat(outputs, -2).matmul(output_).add(output_bias_).add(_input)\r\n");

		public void L2Regularize(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(input, lambda);
			Misc.L2RegularizeIMPL(core, lambda);
			Misc.L2RegularizeIMPL(output_gate, lambda);
			Misc.L2RegularizeIMPL(input_gate_, lambda);
			Misc.L2RegularizeIMPL(output_gate_, lambda);
			Misc.L2RegularizeIMPL(input_gate, lambda);
			Misc.L2RegularizeIMPL(reset_gate, lambda);
			Misc.L2RegularizeIMPL(reset_gate_, lambda);

		}

		public void L1Regularize(Scalar lambda)
		{
			Misc.L1RegularizeIMPL(output, lambda);
		}

		public void L2RegularizeOutput(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(output, lambda);
		}


	}
	public sealed class AOT_KLSTM_Direct : Module<Tensor, Tensor>, IL1Regularizable, IL2Regularizable, IL2Regularizable2
	{

		private readonly Scalar epsilon;

		private readonly Parameter reset_gate_;
		private readonly Parameter reset_gate;
		private readonly Parameter reset_gate_bias;

		private readonly Parameter input_gate_;
		private readonly Parameter input_gate;
		private readonly Parameter input_gate_bias;

		private readonly Parameter output_gate_;
		private readonly Parameter output_gate;
		private readonly Parameter output_gate_bias;

		private readonly Parameter output;
		private readonly Parameter output_bias;




		private static readonly Scalar two = 2.0;
		public AOT_KLSTM_Direct(string name, int size, int coresize, double epsilon) : base(name)
		{
			this.epsilon = epsilon;

			Scalar std = Math.Sqrt(size + coresize);
			using (no_grad())
			{
				input_gate_ = Parameter(randn(size, coresize).div_(std));
				input_gate_bias = Parameter(zeros(coresize));

				reset_gate = Parameter(randn(coresize, coresize).div_(std));
				reset_gate_ = Parameter(randn(size, coresize).div_(std));
				reset_gate_bias = Parameter(zeros(coresize));

				output_gate_ = Parameter(randn(size, coresize).div_(std));

				input_gate = Parameter(randn(coresize, coresize).div_(std));
				output_gate = Parameter(randn(coresize, coresize).div_(std));

				
				output_gate_bias = Parameter(zeros(coresize));
				output_bias = Parameter(zeros(size));

				output = Parameter(randn(coresize, size).div_(Math.Sqrt(size)));
			}



			RegisterComponents();
		}

		public override Tensor forward(Tensor input1)
		{
			if (input1.size(-2) < 2) throw new Exception("Unit or zero length input not accepted!");
			using (Tensor x = compilationUnit.invoke<Tensor>("aot_klstm_core_nodrop_direct", input_gate, reset_gate, output_gate, input_gate_, input_gate_bias, reset_gate_, reset_gate_bias, output_gate_, output_gate_bias, output, output_bias, input1))
			{
				return CustomActivations.Norm(x, epsilon);
			}
		}
		private static readonly jit.CompilationUnit compilationUnit = jit.compile("def aot_klstm_core_nodrop_direct(input_gate : Tensor, reset_gate : Tensor,output_gate : Tensor, input_gate_ : Tensor, input_gate_bias : Tensor, reset_gate_ : Tensor, reset_gate_bias : Tensor, output_gate_ : Tensor, output_gate_bias : Tensor, output_ : Tensor, output_bias_ : Tensor, _input : Tensor) -> Tensor:\r\n    \r\n    length = _input.size(-2)\r\n    \r\n    input_gate_ = _input.matmul(input_gate_).add(input_gate_bias)\r\n    reset_gate_ = _input.transpose(-2,0)[1:length].transpose(-2,0).matmul(reset_gate_).add(reset_gate_bias)\r\n    \r\n    output_gate_ = _input.matmul(output_gate_).add(output_gate_bias)\r\n    \r\n    \r\n    cell_state = _input.select(-2,0).mul(input_gate_.select(-2,0).sigmoid())\r\n    \r\n    hidden_state = cell_state.arctan().mul(output_gate_.select(-2,0).sigmoid()).mul(2.0)\r\n\r\n    \r\n    outputs = [hidden_state.unsqueeze(-2)]\r\n    \r\n    for x in range(1,length):\r\n        gate = cell_state.matmul(input_gate).add(input_gate_.select(-2,x)).sigmoid()\r\n        cell_state = cell_state.mul(reset_gate_.select(-2,x - 1).add(hidden_state.matmul(reset_gate)).sigmoid()).addcmul(_input.select(-2,x), input_gate_.select(-2,x).add(hidden_state.matmul(input_gate)).sigmoid())\r\n        hidden_state = cell_state.arctan().mul(output_gate_.select(-2,x).add(hidden_state.matmul(output_gate)).sigmoid()).mul(2.0)\r\n        outputs.append(hidden_state.unsqueeze(-2))\r\n        pass\r\n    return torch.cat(outputs, -2).matmul(output_).add(output_bias_).add(_input)\r\n");

		public void L2Regularize(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(output_gate, lambda);
			Misc.L2RegularizeIMPL(input_gate_, lambda);
			Misc.L2RegularizeIMPL(output_gate_, lambda);
			Misc.L2RegularizeIMPL(input_gate, lambda);
			Misc.L2RegularizeIMPL(reset_gate, lambda);
			Misc.L2RegularizeIMPL(reset_gate_, lambda);

		}

		public void L1Regularize(Scalar lambda)
		{
			Misc.L1RegularizeIMPL(output, lambda);
		}

		public void L2RegularizeOutput(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(output, lambda);
		}


	}
	public sealed class AOT_SimpleRegularize : autograd.SingleTensorFunction<AOT_SimpleRegularize>
	{
		public override string Name => "AOTSimpleRegularize";
		public AOT_SimpleRegularize(){ regularization = 0.0; }
		private readonly Scalar regularization;
		public AOT_SimpleRegularize(double regularization){
			this.regularization = regularization;
		}

		public override List<Tensor> backward(autograd.AutogradContext ctx, Tensor grad_output)
		{
			return new List<Tensor>() { grad_output.add_(regularization) };
		}

		public override Tensor forward(autograd.AutogradContext ctx, params object[] vars)
		{
			if (vars.Length != 1) throw new Exception("AOT Simple Regularize requires exactly one variable!");
			Tensor t = (Tensor)vars[0];
			ctx.save_for_backward(new List<Tensor>() { t });
			return t.slice(0, 0, t.size(0), 1);
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




}

