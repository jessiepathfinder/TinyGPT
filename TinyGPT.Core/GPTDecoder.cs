﻿using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Parameter = TorchSharp.Modules.Parameter;

namespace TinyGPT.Core
{
	public abstract class FullGPTDecoderUnit : Module<ReadOnlyMemory<ushort>, Tensor>
	{
		protected FullGPTDecoderUnit(string name) : base(name)
		{
		}

		protected FullGPTDecoderUnit(nint handle, nint boxedHandle) : base(handle, boxedHandle)
		{
		}

		public abstract Tensor Forward(ReadOnlySpan<ushort> input);
		public sealed override Tensor forward(ReadOnlyMemory<ushort> input)
		{
			return Forward(input.Span);
		}
	}


	public sealed class GPTDecoderUnitV1 : FullGPTDecoderUnit, IL2Regularizable, IL1Regularizable
	{


		private readonly Parameter wordEmbedding;
		private readonly Parameter staticPositionalEncoding;
		private readonly ModuleList<Module<Tensor, Tensor>> layers = new ModuleList<Module<Tensor, Tensor>>();
		//private readonly ModuleList<TinyRNN> rnnlayers = new ModuleList<TinyRNN>();
		//private readonly ModuleList<ResidualCausalConvolationalLookback> convAttentionLayers = new ModuleList<ResidualCausalConvolationalLookback>();

		private readonly MultiheadSelfAttention finalAttention;
		private readonly ResidualComputeLayer finalCompute;




		private readonly int headcount;
		private readonly Scalar epsilon;

		private readonly int max_context_size;

		//private readonly Linear supplementalEngine;
		//public readonly Tensor supplementalWordEmbedding;
		//private readonly Linear premix;

		public GPTDecoderUnitV1(string name, int latentTokenSize, int attentionHeadsCount, int firstTierAttentionLayers, double positionalEncodingStd, double epsilon, int max_context_size, double wordEmbeddingStd, double initialAttentionGain, double initialComputeGain, double computeDropout, int tokenClasses, double keyQueryInitGain, int attentionSize, double auxAttentionDropout, int grulayers, int gruHiddenStateSize, double gruOutputDropout) : base(name)
		{
			if (attentionHeadsCount < 1)
			{
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}
			//this.pre_expand = Misc.CreateKaimingInitializedLinear(latentTokenSize, pre_expand, false, init.FanInOut.FanIn);
			//expand = Misc.CreateKaimingInitializedLinear(latentTokenSize, multipliedWidth, true, init.FanInOut.FanIn);

			finalCompute = new ResidualComputeLayer("", latentTokenSize, epsilon, initialComputeGain, computeDropout);

			for (int i = 0; i < grulayers; ++i)
			{
				layers.Add(new TinyMGU("", latentTokenSize, gruHiddenStateSize, epsilon, gruOutputDropout));
			}

			for (int i = 0; i < firstTierAttentionLayers; ++i)
			{
				layers.Add(new MultiheadSelfAttention("", latentTokenSize, attentionSize, attentionHeadsCount, epsilon, initialAttentionGain, keyQueryInitGain, auxAttentionDropout));
				layers.Add(new ResidualComputeLayer("", latentTokenSize, epsilon, initialComputeGain, computeDropout));
			}



			//staticPositionalEncoding = Parameter(zeros(max_context_size, latentTokenSize));
			Span<long> wordEmbeddingSize = stackalloc long[2];



			wordEmbeddingSize[1] = latentTokenSize;

			if (positionalEncodingStd > 0.0)
			{
				wordEmbeddingSize[0] = max_context_size;
				staticPositionalEncoding = Parameter(normal(0.0, positionalEncodingStd, wordEmbeddingSize));
			}
			else
			{
				staticPositionalEncoding = Parameter(zeros(max_context_size, latentTokenSize));
			}
			//staticPositionalEncoding = Parameter(randn(max_context_size, latentTokenSize));
			//long ltk = supplementalWordEmbedding.size(0);

			wordEmbeddingSize[0] = tokenClasses;





			//layers.Add(new TinyRNN("", multipliedWidth, multipliedWidth, epsilon));
			finalAttention = new MultiheadSelfAttention("", latentTokenSize, attentionSize, attentionHeadsCount, epsilon, initialAttentionGain, keyQueryInitGain, auxAttentionDropout);
			//finalBias = Parameter(zeros(1, pretrainedWord2Vec.size(0)));
			wordEmbedding = Parameter(wordEmbeddingStd > 0 ? normal(0.0, wordEmbeddingStd, wordEmbeddingSize) : zeros(wordEmbeddingSize));
			headcount = attentionHeadsCount;
			//supplementalEngine = Misc.CreateKaimingInitializedLinear((int)supplementalWordEmbedding.size(1), latentTokenSize, true, init.FanInOut.FanIn);
			RegisterComponents();
			this.epsilon = epsilon;
			this.max_context_size = max_context_size;
		}
		public void FinalForward(ref Tensor x)
		{
			using (Tensor y = x) x = finalCompute.forward(y);
			using (Tensor y = x, x1 = wordEmbedding.transpose(0, 1))
				x = y.matmul(x1);
		}
		public void DecoupledWeightDecayWordEmbeddings(Scalar scalar)
		{
			using (no_grad())
			{
				wordEmbedding.mul_(scalar);
			}
		}
		public Tensor Forward(ReadOnlySpan<ushort> input, int slice, double dropout = 0.0, bool retearly = false)
		{
			int len = input.Length;
			int maxlen2 = max_context_size;
			if (len > maxlen2 | len == 0)
			{
				throw new ArgumentOutOfRangeException(nameof(input));
			}


			long[] longs = new long[len];
			for (int i = 0; i < len; ++i)
			{
				longs[i] = input[i];
			}
			Tensor wordEmbedding = this.wordEmbedding;
			Device device = wordEmbedding.device;
			Scalar epsilon = this.epsilon;


			using (NewDisposeScope())
			{


				Tensor y;
				using (Tensor z2 = tensor(longs, ScalarType.Int64, device, false))
				{
					y = wordEmbedding[z2];
				}


				using (Tensor x = y) if (len == maxlen2)
					{
						y = x.add(staticPositionalEncoding);
					}
					else
					{
						using Tensor z = staticPositionalEncoding.slice(0, 0, len, 1);
						y = x.add(z);
					}
				using (Tensor x = y) y = CustomActivations.Norm(x, epsilon);

				MultiheadSelfAttention finalattention = finalAttention;


				foreach (Module<Tensor, Tensor> hiddenLayer in layers)
				{
					using Tensor x = y;
					if (hiddenLayer is MultiheadSelfAttention sat)
					{
						y = sat.Forward(x, 0, null, dropout, true);
					}
					else
					{
						y = hiddenLayer.forward(x);
					}


				}

				using (Tensor x = y) if (slice > 0)
					{
						using Tensor x2 = y, mask = Transformer.CreateCausalAttentionMask(len - slice, len, ScalarType.Float32, wordEmbedding.device);
						y = finalattention.Forward(x2, slice, mask, dropout);
					}
					else
					{
						y = finalattention.Forward(x, 0, null, dropout, true);
					}
				if (retearly)
				{
					return y.MoveToOuterDisposeScope();
				}
				using (Tensor x = y) y = finalCompute.forward(x);

				/*
				using(yconv){
					using Tensor x = y;
					y = x.add(yconv);
				}
				using (Tensor x = y) y = CustomActivations.Norm(x, epsilon);
				*/
				/*
				if (slice > 0)
				{
					using Tensor x = y;
					y = x.slice(0, slice, len, 1);
				}
				*/


				using (Tensor x = y, x1 = wordEmbedding.transpose(0, 1))
					y = x.matmul(x1);



				return y.MoveToOuterDisposeScope();





			}
		}
		private static readonly Scalar one = 1.0;

		public override Tensor Forward(ReadOnlySpan<ushort> input)
		{
			using (NewDisposeScope())
			{
				using Tensor x = Forward(input, input.Length - 1);
				return x.squeeze(0).MoveToOuterDisposeScope();
			}
		}


		public void L2Regularize(Scalar lambda)
		{
			//Misc.L2RegularizeIMPL(wordEmbedding, lambda);
			foreach (Module<Tensor, Tensor> module in layers)
			{
				if (module is IL2Regularizable regularizable)
				{
					regularizable.L2Regularize(lambda);
				}
			}
			finalAttention.L2Regularize(lambda);
			finalCompute.L2Regularize(lambda);
			Misc.L2RegularizeIMPL(wordEmbedding, lambda);
		}

		public void L1Regularize(Scalar lambda)
		{
			foreach (Module<Tensor, Tensor> module in layers)
			{
				if (module is IL1Regularizable regularizable)
				{
					regularizable.L1Regularize(lambda);
				}
			}
			finalCompute.L1Regularize(lambda);
			Misc.L1RegularizeIMPL(staticPositionalEncoding, lambda);
		}
	}
	public sealed class GPTDecoderUnitV1_2 : FullGPTDecoderUnit, IL2Regularizable, IL1Regularizable, IL2Regularizable2, IHaveSpecialTreatmentLayers
	{




		public readonly Tensor wordEmbeddings;
		private readonly ModuleList<Module<Tensor, Tensor>> layers = new ModuleList<Module<Tensor, Tensor>>();
		//private readonly ModuleList<TinyRNN> rnnlayers = new ModuleList<TinyRNN>();
		//private readonly ModuleList<ResidualCausalConvolationalLookback> convAttentionLayers = new ModuleList<ResidualCausalConvolationalLookback>();

		private readonly MultiheadSelfAttention finalAttention;
		private readonly ResidualComputeLayer2 finalCompute;
		//private readonly Parameter finalLinear;


		private readonly int headcount;
		private readonly Scalar epsilon;

		private readonly int unorderedConvKernelSize;
		public readonly Parameter finalLinear;

		//private readonly Linear supplementalEngine;
		//public readonly Tensor supplementalWordEmbedding;

		//private readonly TinyMGU finalMGU;
		public void FinalForward(ref Tensor x)
		{
			using (Tensor y = x) x = finalCompute.forward(y);
			//using (Tensor y = x) x = y.matmul(finalLinear);
			using (Tensor y = x)
				x = y.matmul(finalLinear);
		}
		public Tensor FinalForward2(Tensor x) => x.matmul(finalLinear);
		public void Hack1(){
			finalLinear.requires_grad = false;
		}
		public void RReinit(){
			using(no_grad()){
				Tensor we = wordEmbeddings;
				long lts = we.size(1);
				long tcz = we.size(0);
				Tensor x;
				using (Tensor ranmul = randn(lts, lts, we.dtype, we.device)){
					ranmul.div_(Math.Sqrt(lts * tcz));
					x = we.matmul(ranmul);
				}
				using(x){
					x.transpose_(0, 1);
					finalLinear.copy_(x);

				}

			}
		}
		public GPTDecoderUnitV1_2(string name, int latentTokenSize, int attentionHeadsCount, int firstTierAttentionLayers, double epsilon, double initialAttentionGain, double initialComputeGain, double computeDropout, int tokenClasses, double keyQueryInitGain, int attentionSize, double auxAttentionDropout, int lstmlayers, int lstmHiddenStateSize, double lstmOutputDropout, int unorderedConvKernelSize, Tensor pretrainedword2vec, bool useAlternateInitialization, bool usePeepholeKLSTM, int directKLSTMLayers) : base(name)
		{
			if (attentionHeadsCount < 1)
			{
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}
			//this.pre_expand = Misc.CreateKaimingInitializedLinear(latentTokenSize, pre_expand, false, init.FanInOut.FanIn);
			//expand = Misc.CreateKaimingInitializedLinear(latentTokenSize, multipliedWidth, true, init.FanInOut.FanIn);

			finalCompute = new ResidualComputeLayer2("", latentTokenSize, epsilon, initialComputeGain, computeDropout);
			for (int i = 0; i < directKLSTMLayers; ++i)
			{
				layers.Add(new AOT_KLSTM_Direct("", latentTokenSize, lstmHiddenStateSize, epsilon));
			}
			if (usePeepholeKLSTM){
				for (int i = 0; i < lstmlayers; ++i)
				{
					layers.Add(new AOT_KLSTM_Peephole("", latentTokenSize, lstmHiddenStateSize, epsilon));
				}
			} else{
				for (int i = 0; i < lstmlayers; ++i)
				{
					layers.Add(new AOT_KLSTM("", latentTokenSize, lstmHiddenStateSize, epsilon, lstmOutputDropout));
				}
			}
			for (int i = 0; i < firstTierAttentionLayers; ++i)
			{
				layers.Add(new MultiheadSelfAttention("", latentTokenSize, attentionSize, attentionHeadsCount, epsilon, initialAttentionGain, keyQueryInitGain, auxAttentionDropout));
				layers.Add(new ResidualComputeLayer2("", latentTokenSize, epsilon, initialComputeGain, computeDropout));
			}


			//finalMGU = new TinyMGU("", latentTokenSize, gruHiddenStateSize, epsilon, gruOutputDropout);

			//staticPositionalEncoding = Parameter(zeros(max_context_size, latentTokenSize));
			Span<long> wordEmbeddingSize = stackalloc long[2];


			wordEmbeddingSize[1] = tokenClasses;

			
			//staticPositionalEncoding = Parameter(randn(max_context_size, latentTokenSize));
			//long ltk = supplementalWordEmbedding.size(0);

			wordEmbeddingSize[0] = latentTokenSize;
			//init.normal_(preconv.weight ?? throw new Exception("Conv had no weight (should not reach here)"), 0.0, 1.0 / Math.Sqrt(latentTokenSize * preAttentionKernelSize));
			this.wordEmbeddings = pretrainedword2vec;

			


			//layers.Add(new TinyRNN("", multipliedWidth, multipliedWidth, epsilon));
			finalAttention = new MultiheadSelfAttention("", latentTokenSize, attentionSize, attentionHeadsCount, epsilon, initialAttentionGain, keyQueryInitGain, auxAttentionDropout);
			//finalBias = Parameter(zeros(1, pretrainedWord2Vec.size(0)));
			//singleLayerPerceptron = Parameter(wordEmbeddingStd > 0 ? normal(0.0, wordEmbeddingStd, wordEmbeddingSize) : zeros(wordEmbeddingSize));
			headcount = attentionHeadsCount;
			//supplementalEngine = Misc.CreateKaimingInitializedLinear((int)supplementalWordEmbedding.size(1), latentTokenSize, true, init.FanInOut.FanIn);

			//byte[,] identity = new byte[latentTokenSize, latentTokenSize];
			//for (int i = 0; i < latentTokenSize; ++i) identity[i, i] = 1;
			Tensor fl;
			if(useAlternateInitialization){
				double rsqt = Math.Sqrt(3.0 / tokenClasses);
				fl = init.uniform_(empty(latentTokenSize, tokenClasses), -rsqt, rsqt);
			} else{
				fl = randn(latentTokenSize, tokenClasses).div_(Math.Sqrt(tokenClasses));
			}
			finalLinear = Parameter(fl);
			RegisterComponents();
			this.epsilon = epsilon;
			this.unorderedConvKernelSize = unorderedConvKernelSize;
		}

		public Tensor Forward(ReadOnlySpan<ushort> input, int slice, double dropout = 0.0,bool retearly = false,bool retearly2 = false)
		{
			int len = input.Length;

			bool multioutput = (len - slice) > 1;

			int softslice = input.IndexOf((ushort)0);
			bool hassoftslice = softslice > -1;


			long[] longs = new long[len];
			for (int i = 0; i < len; ++i)
			{
				longs[i] = input[i];
			}
			Tensor pretrainedword2vec = wordEmbeddings;
			Device device = pretrainedword2vec.device;
			Scalar epsilon = this.epsilon;
			int unorderedConvKernelSize = this.unorderedConvKernelSize;
			int km1 = unorderedConvKernelSize - 1;

			using (NewDisposeScope())
			{


				Tensor y;
				using (Tensor z2 = tensor(longs, ScalarType.Int64, device, false))
				{
					y = pretrainedword2vec[z2];
				}

				if(km1 > 0){
					using (Tensor x = y) y = x.transpose(0, 1);

					{
						Tensor? remerge;
						if (hassoftslice)
						{
							using Tensor x = y;
							remerge = x.slice(1, 0, softslice, 1);
							y = x.slice(1, softslice, len, 1);
						}
						else
						{
							remerge = null;
						}
						using (Tensor x = y) y = functional.pad(x, (km1, 0), PaddingModes.Zeros, 0.0);
						using (Tensor x = y) y = functional.avg_pool1d(x, unorderedConvKernelSize, 1, 0, false, false);
						if (remerge is { })
						{
							using (Tensor x = remerge) remerge = functional.pad(x, (km1, 0), PaddingModes.Zeros, 0.0);
							using (Tensor x = remerge) remerge = functional.avg_pool1d(x, unorderedConvKernelSize, 1, 0, false, false);
							using (remerge)
							{
								using Tensor x = y;
								y = cat(new Tensor[] { remerge, x }, 1);
							}
						}
					}
					using (Tensor x = y) y = x.transpose(0, 1);
				}
				using (Tensor x = y) y = CustomActivations.Norm(x, epsilon);

				Tensor? mask = hassoftslice ? Transformer.CreateBARTAttentionMask(len, softslice, ScalarType.Float32, device) : null;


				foreach (Module<Tensor, Tensor> hiddenLayer in layers)
				{

					using Tensor x = y;
					if (hiddenLayer is MultiheadSelfAttention sat)
					{
						
						y = sat.Forward(x, 0, mask, dropout, false);
					}
					else
					{
						y = hiddenLayer.forward(x);
					}


				}

				if(multioutput){
					if(mask is { })
					{
						using Tensor x = mask;
						mask = x.slice(0, slice, len, 1);
					}
				} else if(mask is { }){
					mask.Dispose();
					mask = null;
				}

				
				using (Tensor x = y) y = finalAttention.Forward(x, slice, mask, dropout, false);
				mask?.Dispose();

				if (retearly) return y.MoveToOuterDisposeScope();


				using (Tensor x = y) y = finalCompute.forward(x);
				if(retearly2) return y.MoveToOuterDisposeScope();
				//using (Tensor x = y) y = x.matmul(finalLinear);
				using (Tensor x = y)y = x.matmul(finalLinear);



				return y.MoveToOuterDisposeScope();





			}
		}
		private static readonly Scalar one = 1.0;

		public override Tensor Forward(ReadOnlySpan<ushort> input)
		{
			using (NewDisposeScope())
			{
				using Tensor x = Forward(input, input.Length - 1);
				return x.squeeze(0).MoveToOuterDisposeScope();
			}
		}


		public void L2Regularize(Scalar lambda)
		{

			foreach (Module<Tensor, Tensor> module in layers)
			{
				if (module is IL2Regularizable regularizable)
				{
					regularizable.L2Regularize(lambda);
				}
			}
			finalAttention.L2Regularize(lambda);
			finalCompute.L2Regularize(lambda);
			Misc.L2RegularizeIMPL(finalLinear, lambda);
		}

		public void L1Regularize(Scalar lambda)
		{

			foreach (Module<Tensor, Tensor> module in layers)
			{
				if (module is IL1Regularizable regularizable)
				{
					regularizable.L1Regularize(lambda);
				}
			}
			finalCompute.L1Regularize(lambda);
			//Misc.L1RegularizeIMPL(preconv.weight, lambda);

		}

		public void L2RegularizeOutput(Scalar lambda)
		{
			foreach (Module<Tensor, Tensor> module in layers)
			{
				if (module is IL2Regularizable2 regularizable)
				{
					regularizable.L2RegularizeOutput(lambda);
				}
			}
			//finalAttention.L2RegularizeOutput(lambda);
			finalCompute.L2RegularizeOutput(lambda);
		}

		public IEnumerable<Parameter> GetSpecialTreatmentLayers()
		{
			foreach(Module<Tensor, Tensor> module in layers){
				if(module is IHaveSpecialTreatmentLayers haveSpecialTreatmentLayers){
					foreach (Parameter p in haveSpecialTreatmentLayers.GetSpecialTreatmentLayers()) yield return p;
				}
			}
			foreach (Parameter p in finalAttention.GetSpecialTreatmentLayers()) yield return p;
		}
	}
	public sealed class GPTDecoderUnitV1_1 : FullGPTDecoderUnit, IL2Regularizable, IL1Regularizable
	{


		private readonly Parameter wordEmbedding;
		private readonly Linear unordered_conv;
		private readonly Parameter staticPositionalEncoding;
		private readonly Parameter staticInputPositionalEncoding;
		private readonly Linear mixer;

		private readonly ModuleList<KLSTM> preattention = new ModuleList<KLSTM>();
		private readonly ModuleList<Module<Tensor, Tensor>> layers = new ModuleList<Module<Tensor, Tensor>>();
		//private readonly ModuleList<TinyRNN> rnnlayers = new ModuleList<TinyRNN>();
		//private readonly ModuleList<ResidualCausalConvolationalLookback> convAttentionLayers = new ModuleList<ResidualCausalConvolationalLookback>();

		private readonly MultiheadSelfAttention finalAttention;
		private readonly ResidualComputeLayer2 finalCompute;



		private readonly int headcount;
		private readonly Scalar epsilon;

		private readonly int max_context_size;
		private readonly int unorderedConvKernelSize;

		//private readonly Linear supplementalEngine;
		//public readonly Tensor supplementalWordEmbedding;

		//private readonly TinyMGU finalMGU;
		public void FinalForward(ref Tensor x)
		{
			using (Tensor y = x) x = finalCompute.forward(y);
			using (Tensor y = x, x1 = wordEmbedding.transpose(0, 1))
				x = y.matmul(x1);
		}
		public GPTDecoderUnitV1_1(string name, int latentTokenSize, int attentionHeadsCount, int firstTierAttentionLayers, double positionalEncodingStd, double epsilon, int max_context_size, double wordEmbeddingStd, double initialAttentionGain, double initialComputeGain, double computeDropout, int tokenClasses, double keyQueryInitGain, int attentionSize, double auxAttentionDropout, int grulayers, int gruHiddenStateSize, double gruOutputDropout, int unorderedConvKernelSize) : base(name)
		{
			if (attentionHeadsCount < 1)
			{
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}
			//this.pre_expand = Misc.CreateKaimingInitializedLinear(latentTokenSize, pre_expand, false, init.FanInOut.FanIn);
			//expand = Misc.CreateKaimingInitializedLinear(latentTokenSize, multipliedWidth, true, init.FanInOut.FanIn);

			finalCompute = new ResidualComputeLayer2("", latentTokenSize, epsilon, initialComputeGain, computeDropout);

			for (int i = 0; i < grulayers; ++i)
			{
				preattention.Add(new KLSTM("", latentTokenSize, gruHiddenStateSize, epsilon, gruOutputDropout));
			}
			for (int i = 0; i < firstTierAttentionLayers; ++i)
			{
				layers.Add(new MultiheadSelfAttention("", latentTokenSize, attentionSize, attentionHeadsCount, epsilon, initialAttentionGain, keyQueryInitGain, auxAttentionDropout));
				layers.Add(new ResidualComputeLayer2("", latentTokenSize, epsilon, initialComputeGain, computeDropout));
			}
			unordered_conv = Misc.CreateZeroInitializedLinear(latentTokenSize, latentTokenSize, true);


			//finalMGU = new TinyMGU("", latentTokenSize, gruHiddenStateSize, epsilon, gruOutputDropout);

			//staticPositionalEncoding = Parameter(zeros(max_context_size, latentTokenSize));
			Span<long> wordEmbeddingSize = stackalloc long[2];

			mixer = Misc.CreateKaimingInitializedLinear(latentTokenSize, latentTokenSize, true, init.FanInOut.FanIn, 1.0);

			wordEmbeddingSize[1] = latentTokenSize;

			if (positionalEncodingStd > 0.0)
			{
				wordEmbeddingSize[0] = max_context_size - 1;
				staticPositionalEncoding = Parameter(normal(0.0, positionalEncodingStd, wordEmbeddingSize));
				wordEmbeddingSize[0] = max_context_size;
				staticInputPositionalEncoding = Parameter(normal(0.0, positionalEncodingStd, wordEmbeddingSize));
			}
			else
			{
				staticPositionalEncoding = Parameter(zeros(max_context_size - 1, latentTokenSize));
				staticInputPositionalEncoding = Parameter(zeros(max_context_size, latentTokenSize));
			}
			//staticPositionalEncoding = Parameter(randn(max_context_size, latentTokenSize));
			//long ltk = supplementalWordEmbedding.size(0);

			wordEmbeddingSize[0] = tokenClasses;
			//init.normal_(preconv.weight ?? throw new Exception("Conv had no weight (should not reach here)"), 0.0, 1.0 / Math.Sqrt(latentTokenSize * preAttentionKernelSize));




			//layers.Add(new TinyRNN("", multipliedWidth, multipliedWidth, epsilon));
			finalAttention = new MultiheadSelfAttention("", latentTokenSize, attentionSize, attentionHeadsCount, epsilon, initialAttentionGain, keyQueryInitGain, auxAttentionDropout);
			//finalBias = Parameter(zeros(1, pretrainedWord2Vec.size(0)));
			wordEmbedding = Parameter(wordEmbeddingStd > 0 ? normal(0.0, wordEmbeddingStd, wordEmbeddingSize) : zeros(wordEmbeddingSize));
			headcount = attentionHeadsCount;
			//supplementalEngine = Misc.CreateKaimingInitializedLinear((int)supplementalWordEmbedding.size(1), latentTokenSize, true, init.FanInOut.FanIn);
			RegisterComponents();
			this.epsilon = epsilon;
			this.max_context_size = max_context_size;
			this.unorderedConvKernelSize = unorderedConvKernelSize;
		}

		public Tensor Forward(ReadOnlySpan<ushort> input, int slice, double dropout = 0.0, bool retearly = false, bool retearly2 = false)
		{
			int len = input.Length;
			int maxlen2 = max_context_size;
			if (len > maxlen2 | len == 0)
			{
				throw new ArgumentOutOfRangeException(nameof(input));
			}
			bool multioutput = (len - slice) > 1;

			int softslice = input.IndexOf((ushort)0);
			bool hassoftslice = softslice > -1;


			long[] longs = new long[len];
			for (int i = 0; i < len; ++i)
			{
				longs[i] = input[i];
			}
			Tensor wordEmbedding = this.wordEmbedding;
			Device device = wordEmbedding.device;
			Scalar epsilon = this.epsilon;
			Tensor[] tl = new Tensor[2];
			int unorderedConvKernelSize = this.unorderedConvKernelSize;

			using (NewDisposeScope())
			{


				Tensor y;
				using (Tensor z2 = tensor(longs, ScalarType.Int64, device, false))
				{
					y = wordEmbedding[z2];
				}

				Tensor cnv = y.transpose(0, 1);
				using (Tensor x = cnv) cnv = functional.pad(x, (unorderedConvKernelSize - 1, 0), PaddingModes.Zeros, 0.0);
				using (Tensor x = cnv) cnv = functional.avg_pool1d(x, unorderedConvKernelSize, 1, 0, false, false);
				using (Tensor x = cnv) cnv = x.transpose(0, 1);
				using (Tensor x = cnv) cnv = CustomActivations.Norm(x, epsilon);
				using (Tensor x = cnv) cnv = unordered_conv.forward(x);

				Tensor lat = CustomActivations.Norm(y, epsilon);
				using (cnv)
				{
					using Tensor x = lat;
					lat = x.add(cnv);
				}
				using (Tensor x = lat) lat = CustomActivations.Norm(x, epsilon);
				foreach (KLSTM tinyMGU in preattention)
				{
					using Tensor x = lat;
					lat = tinyMGU.forward(x);
				}

				Tensor? mask;
				if (hassoftslice)
				{
					Tensor ss1;
					if (softslice == maxlen2)
					{
						using Tensor x = y;
						y = x.add(staticInputPositionalEncoding);
					}
					else
					{
						using (Tensor z = staticInputPositionalEncoding.slice(0, 0, softslice, 1), x = y.slice(0, 0, softslice, 1))
							ss1 = x.add(z);
						using (Tensor x = y) y = x.slice(0, softslice, len, 1);

						if (softslice == 0 & len == maxlen2)
						{
							using Tensor x = y;
							y = x.add(staticPositionalEncoding);
						}
						else
						{
							using Tensor z = staticPositionalEncoding.slice(0, 0, len - softslice, 1), x = y;
							y = x.add(z);
						}
						using (ss1)
						{
							using Tensor x = y;
							tl[0] = ss1;
							tl[1] = x;
							y = cat(tl, 0);
						}
					}
					mask = Transformer.CreateBARTAttentionMask(len, softslice, ScalarType.Float32, device);
				}
				else
				{
					using Tensor x = y;
					if (len == maxlen2)
					{
						y = x.add(staticInputPositionalEncoding);
					}
					else
					{
						using Tensor z = staticInputPositionalEncoding.slice(0, 0, len, 1);
						y = x.add(z);
					}
					mask = null;
				}







				using (Tensor x = y) y = CustomActivations.Norm(x, epsilon);
				using (Tensor x = y) y = mixer.forward(x);
				using (lat)
				{
					using Tensor x = y;
					y = x.add(lat);
				}

				using (Tensor x = y) y = CustomActivations.Norm(x, epsilon);




				foreach (Module<Tensor, Tensor> hiddenLayer in layers)
				{

					using Tensor x = y;
					if (hiddenLayer is MultiheadSelfAttention sat)
					{

						y = sat.Forward(x, 0, mask, dropout, false);
					}
					else
					{
						y = hiddenLayer.forward(x);
					}


				}

				if (multioutput)
				{
					if (mask is { })
					{
						using Tensor x = mask;
						mask = x.slice(0, slice, len, 1);
					}
				}
				else if (mask is { })
				{
					mask.Dispose();
					mask = null;
				}


				using (Tensor x = y) y = finalAttention.Forward(x, slice, mask, dropout, false);
				mask?.Dispose();

				if (retearly) return y.MoveToOuterDisposeScope();


				using (Tensor x = y) y = finalCompute.forward(x);
				if (retearly2) return y.MoveToOuterDisposeScope();

				using (Tensor x = y, x1 = wordEmbedding.transpose(0, 1))
					y = x.matmul(x1);



				return y.MoveToOuterDisposeScope();





			}
		}
		private static readonly Scalar one = 1.0;

		public override Tensor Forward(ReadOnlySpan<ushort> input)
		{
			using (NewDisposeScope())
			{
				using Tensor x = Forward(input, input.Length - 1);
				return x.squeeze(0).MoveToOuterDisposeScope();
			}
		}


		public void L2Regularize(Scalar lambda)
		{
			//Misc.L2RegularizeIMPL(wordEmbedding, lambda);
			foreach (KLSTM module in preattention)
			{
				module.L2Regularize(lambda);
			}
			foreach (Module<Tensor, Tensor> module in layers)
			{
				if (module is IL2Regularizable regularizable)
				{
					regularizable.L2Regularize(lambda);
				}
			}
			finalAttention.L2Regularize(lambda);
			finalCompute.L2Regularize(lambda);
			Misc.L2RegularizeIMPL(wordEmbedding, lambda);
			Misc.L2RegularizeIMPL(mixer.weight, lambda);
			Misc.L2RegularizeIMPL(unordered_conv.weight, lambda);
		}

		public void L1Regularize(Scalar lambda)
		{
			foreach (KLSTM module in preattention)
			{
				module.L1Regularize(lambda);
			}
			foreach (Module<Tensor, Tensor> module in layers)
			{
				if (module is IL1Regularizable regularizable)
				{
					regularizable.L1Regularize(lambda);
				}
			}
			finalCompute.L1Regularize(lambda);
			//Misc.L1RegularizeIMPL(preconv.weight, lambda);
			Misc.L1RegularizeIMPL(staticPositionalEncoding, lambda);
			Misc.L1RegularizeIMPL(staticInputPositionalEncoding, lambda);
		}
	}
	public sealed class GPTDecoderUnitV1_3 : FullGPTDecoderUnit, IL2Regularizable, IL1Regularizable, IL2Regularizable2, IHaveSpecialTreatmentLayers
	{




		private readonly ModuleList<Module<Tensor, Tensor>> layers = new ModuleList<Module<Tensor, Tensor>>();
		//private readonly ModuleList<TinyRNN> rnnlayers = new ModuleList<TinyRNN>();
		//private readonly ModuleList<ResidualCausalConvolationalLookback> convAttentionLayers = new ModuleList<ResidualCausalConvolationalLookback>();

		private readonly MultiheadSelfAttention finalAttention;
		private readonly ResidualComputeLayer2 finalCompute;
		//private readonly Parameter finalLinear;


		private readonly int headcount;
		private readonly Scalar epsilon;
		private readonly InitDecoder initDecoder;
		public readonly Parameter finalLinear;

		//private readonly Linear supplementalEngine;
		//public readonly Tensor supplementalWordEmbedding;

		//private readonly TinyMGU finalMGU;
		
		public Tensor FinalForward2(Tensor x) => x.matmul(finalLinear);
		private readonly int causalConvPaddingSize;
		private readonly int multicompute;
		public GPTDecoderUnitV1_3(string name, int latentTokenSize, int attentionHeadsCount, int firstTierAttentionLayers, double epsilon, double initialAttentionGain, double initialComputeGain, double computeDropout, int tokenClasses, double keyQueryInitGain, int attentionSize, double auxAttentionDropout, int lstmlayers, int lstmHiddenStateSize, double lstmOutputDropout, InitDecoder initDecoder, int multiApplyComputeLayers) : base(name)
		{
			if (attentionHeadsCount < 1)
			{
				throw new ArgumentNullException(nameof(attentionHeadsCount));
			}
			//this.pre_expand = Misc.CreateKaimingInitializedLinear(latentTokenSize, pre_expand, false, init.FanInOut.FanIn);
			//expand = Misc.CreateKaimingInitializedLinear(latentTokenSize, multipliedWidth, true, init.FanInOut.FanIn);
			this.initDecoder = initDecoder;
			finalCompute = new ResidualComputeLayer2("", latentTokenSize, epsilon, initialComputeGain, computeDropout);
			for (int i = 0; i < lstmlayers; ++i)
			{
				layers.Add(new AOT_KLSTM_Bugfix("", latentTokenSize, lstmHiddenStateSize, epsilon, lstmOutputDropout));
			}
			for (int i = 0; i < firstTierAttentionLayers; ++i)
			{
				layers.Add(new MultiheadSelfAttention("", latentTokenSize, attentionSize, attentionHeadsCount, epsilon, initialAttentionGain, keyQueryInitGain, auxAttentionDropout));
				layers.Add(new ResidualComputeLayer2("", latentTokenSize, epsilon, initialComputeGain, computeDropout));
			}


			//finalMGU = new TinyMGU("", latentTokenSize, gruHiddenStateSize, epsilon, gruOutputDropout);

			//staticPositionalEncoding = Parameter(zeros(max_context_size, latentTokenSize));
			Span<long> wordEmbeddingSize = stackalloc long[2];


			wordEmbeddingSize[1] = tokenClasses;



			wordEmbeddingSize[0] = latentTokenSize;




			finalAttention = new MultiheadSelfAttention("", latentTokenSize, attentionSize, attentionHeadsCount, epsilon, initialAttentionGain, keyQueryInitGain, auxAttentionDropout);
			headcount = attentionHeadsCount;


			finalLinear = Parameter(randn(latentTokenSize, tokenClasses).div_(Math.Sqrt(tokenClasses)));

			RegisterComponents();
			this.epsilon = epsilon;
			multicompute = multiApplyComputeLayers;
		}

		public Tensor Forward(ReadOnlySpan<ushort> input, int slice, double dropout = 0.0)
		{
			int len = input.Length;
			int map = multicompute;

			bool multioutput = (len - slice) > 1;

			int softslice = input.IndexOf((ushort)0);
			if (softslice < 0) throw new Exception("[START_GPT] token is required by this model!");
			if (slice < softslice) throw new Exception("Output sliced before [START_GPT] token!");

			long[] longs = new long[len];
			for (int i = 0; i < len; ++i)
			{
				longs[i] = input[i];
			}
			InitDecoder initDecoder = this.initDecoder;
			Tensor pretrainedword2vec = initDecoder.wordEmbeddings;
			Device device = pretrainedword2vec.device;
			Scalar epsilon = this.epsilon;
			bool sl0 = slice > 0;
			bool zss = softslice == 0;



			using (NewDisposeScope())
			{


				Tensor y;
				using (Tensor z2 = tensor(longs, ScalarType.Int64, device, false))
				{
					y = pretrainedword2vec[z2];
				}


				Tensor ct;

				if(zss){
					ct = y;
					y = CustomActivations.Norm(y, epsilon);
					initDecoder.DoIt(ref ct, slice - softslice);
				} else{
					ct = y.slice(0, softslice, len, 1);
					initDecoder.DoIt(ref ct, slice - softslice);
					using Tensor x = y; y = CustomActivations.Norm(x, epsilon);
				}
				



				

				Tensor? mask = multioutput ? Transformer.CreateBARTAttentionMask(len, softslice, ScalarType.Float32, device) : null;


				foreach (Module<Tensor, Tensor> hiddenLayer in layers)
				{
					if(hiddenLayer is ResidualComputeLayer2 residualComputeLayer){
						for(int i = 0; i < map; ++i){
							using Tensor x1 = y;
							y = residualComputeLayer.forward(x1);
						}
						continue;
					}
					using Tensor x = y;
					if (hiddenLayer is MultiheadSelfAttention sat)
					{

						y = sat.Forward(x, 0, mask, dropout, false);
					}
					else
					{
						y = hiddenLayer.forward(x);
					}


				}

				if (multioutput & sl0)
				{
					if (mask is { })
					{
						using Tensor x = mask;
						mask = x.slice(0, slice, len, 1);
					}
				}
				else if (mask is { })
				{
					mask.Dispose();
					mask = null;
				}


				using (Tensor x = y) y = finalAttention.Forward(x, slice, mask, dropout, false);
				mask?.Dispose();



				for(int i = 0; i < map;++i) using (Tensor x = y) y = finalCompute.forward(x);

				using (Tensor x = y) y = CustomActivations.Norm(x, epsilon);
				using (Tensor x = y) y = x.matmul(finalLinear);
				using (ct) using (Tensor x = y) y = x.add(ct);
				

				return y.MoveToOuterDisposeScope();





			}
		}
		private static readonly Scalar one = 1.0;

		public override Tensor Forward(ReadOnlySpan<ushort> input)
		{
			using (NewDisposeScope())
			{
				using Tensor x = Forward(input, input.Length - 1);
				return x.squeeze(0).MoveToOuterDisposeScope();
			}
		}


		public void L2Regularize(Scalar lambda)
		{

			foreach (Module<Tensor, Tensor> module in layers)
			{
				if (module is IL2Regularizable regularizable)
				{
					regularizable.L2Regularize(lambda);
				}
			}
			finalAttention.L2Regularize(lambda);
			finalCompute.L2Regularize(lambda);
			Misc.L2RegularizeIMPL(finalLinear, lambda);
		}
		public void L2RegularizeSkipAttn(Scalar lambda)
		{

			foreach (Module<Tensor, Tensor> module in layers)
			{
				if (module is IL2Regularizable regularizable)
				{
					if (module is MultiheadSelfAttention) continue;
					regularizable.L2Regularize(lambda);
				}
			}
			//finalAttention.L2Regularize(lambda);
			finalCompute.L2Regularize(lambda);
			Misc.L2RegularizeIMPL(finalLinear, lambda);
		}

		public void L1Regularize(Scalar lambda)
		{

			foreach (Module<Tensor, Tensor> module in layers)
			{
				if (module is IL1Regularizable regularizable)
				{
					regularizable.L1Regularize(lambda);
				}
			}
			finalCompute.L1Regularize(lambda);
			//Misc.L1RegularizeIMPL(preconv.weight, lambda);

		}

		public void L2RegularizeOutput(Scalar lambda)
		{
			foreach (Module<Tensor, Tensor> module in layers)
			{
				if (module is IL2Regularizable2 regularizable)
				{
					regularizable.L2RegularizeOutput(lambda);
				}
			}
			//finalAttention.L2RegularizeOutput(lambda);
			finalCompute.L2RegularizeOutput(lambda);
		}

		public IEnumerable<Parameter> GetSpecialTreatmentLayers()
		{
			foreach (Module<Tensor, Tensor> module in layers)
			{
				if (module is IHaveSpecialTreatmentLayers haveSpecialTreatmentLayers)
				{
					foreach (Parameter p in haveSpecialTreatmentLayers.GetSpecialTreatmentLayers()) yield return p;
				}
			}
			foreach (Parameter p in finalAttention.GetSpecialTreatmentLayers()) yield return p;
		}
		public void PrepareRetrain()
		{
			Tensor o = finalLinear;
			o.zero_();
			o.requires_grad = true;
			
			initDecoder.PrepareRetrain();
		}
	}
	public sealed class InitDecoder : FullGPTDecoderUnit, IL2Regularizable
	{
		public readonly Tensor wordEmbeddings;
		private readonly Parameter input;
		private readonly Parameter input_bias;
		private readonly Parameter core;

		private readonly Parameter output;

		private static readonly jit.CompilationUnit compilationUnit = jit.compile("def aot_rnn_core_nodrop_special(core : Tensor, input_ : Tensor, slicey: int) -> Tensor:    \r\n    length = input_.size(-2)    \r\n    hidden_state = input_.select(-2,0).arctan()\r\n    if(length == 1):\r\n        return hidden_state.unsqueeze(-2)\r\n    \r\n    outputs = []\r\n    if slicey == 0:\r\n        outputs.append(hidden_state.unsqueeze(-2))\r\n    \r\n    for x in range(1,length):\r\n        hidden_state = input_.select(-2,x).add(hidden_state.matmul(core)).arctan()\r\n        if x >= slicey:\r\n            outputs.append(hidden_state.unsqueeze(-2))\r\n\r\n    return torch.cat(outputs, -2) if (length - slicey) > 1 else outputs[0]\r\n");
		public void PrepareRetrain(){
			Tensor o = output;
			o.zero_();
			o.requires_grad = true;
			
		}
		public void Hack2(){
			input.requires_grad = false;
			input_bias.requires_grad = false;
			core.requires_grad = false;
		}
		public InitDecoder(string name, Tensor wordEmbeddings, long memorySize, long convKernelSize, long tokenClasses) : base(name)
		{
			this.wordEmbeddings = wordEmbeddings;
			long lts = wordEmbeddings.size(1);
			Scalar div = Math.Sqrt((lts * convKernelSize) + memorySize);
			input = nn.Parameter(randn(memorySize, lts, convKernelSize).div_(div));
			input_bias = nn.Parameter(zeros(memorySize));
			core = nn.Parameter(randn(memorySize, memorySize).div_(div));
			output = nn.Parameter(randn(memorySize, tokenClasses).div_(Math.Sqrt(tokenClasses)));
			RegisterComponents();
		}

		public void DoIt(ref Tensor x, int slice){
			Tensor cw = input;

			using (Tensor y = x) x = y.transpose(-2, -1);
			using (Tensor y = x) x = functional.pad(y, (cw.size(2) - 1, 0), PaddingModes.Zeros, 0.0);
			using (Tensor y = x) x = functional.conv1d(y, cw, input_bias);

			using (Tensor y = x) x = y.transpose(-2, -1);

			using (Tensor y = x) x = compilationUnit.invoke<Tensor>("aot_rnn_core_nodrop_special", core, y, slice);
			using (Tensor y = x) x = y.matmul(output);
		}

		public override Tensor Forward(ReadOnlySpan<ushort> input)
		{
			using (NewDisposeScope())
			{
				using Tensor x = Forward(input, input.Length - 1);
				return x.squeeze(0).MoveToOuterDisposeScope();
			}
		}
		public Tensor Forward(ReadOnlySpan<ushort> input, int slice)
		{
			int len = input.Length;
			if (slice >= len | slice < 0) throw new Exception("Invalid slice!");
			long[] longarr = new long[len];
			for (int i = 0; i < len; ++i) longarr[i] = input[i];
			Tensor we = wordEmbeddings;
			using (NewDisposeScope()){
				Tensor y;

				using (Tensor x = tensor(longarr, int64, we.device)) y = we[x];
				DoIt(ref y, slice);
				return y.MoveToOuterDisposeScope();
			}
		}

		public void L2Regularize(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(input, lambda);
			Misc.L2RegularizeIMPL(core, lambda);
			Misc.L2RegularizeIMPL(output, lambda);
		}
		public void L2RegularizeOutput(Scalar lambda)
		{
			Misc.L2RegularizeIMPL(output, lambda);
		}
	}
}