
using System.ComponentModel;
using TorchSharp;
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
			finalAttention.L1Regularize(lambda);
			finalCompute.L1Regularize(lambda);
			Misc.L1RegularizeIMPL(staticPositionalEncoding, lambda);
		}
	}
	public sealed class GPTDecoderUnitV1_2 : FullGPTDecoderUnit, IL2Regularizable, IL1Regularizable
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
		private readonly Parameter finalLinear;

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
		public GPTDecoderUnitV1_2(string name, int latentTokenSize, int attentionHeadsCount, int firstTierAttentionLayers, double epsilon, double initialAttentionGain, double initialComputeGain, double computeDropout, int tokenClasses, double keyQueryInitGain, int attentionSize, double auxAttentionDropout, int grulayers, int gruHiddenStateSize, double gruOutputDropout, int unorderedConvKernelSize, Tensor pretrainedword2vec) : base(name)
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
				layers.Add(new AOT_KLSTM("", latentTokenSize, gruHiddenStateSize, epsilon, gruOutputDropout));
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

			finalLinear = Parameter(randn(latentTokenSize,tokenClasses).div_(Math.Sqrt(tokenClasses)));
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

				using (Tensor x = y) y = x.transpose(0, 1);

				{
					Tensor? remerge;
					if(hassoftslice){
						using Tensor x = y;
						remerge = x.slice(1, 0, softslice, 1);
						y = x.slice(1, softslice, len, 1);
					} else{
						remerge = null;
					}
					using (Tensor x = y) y = functional.pad(x, (km1, 0), PaddingModes.Zeros, 0.0);
					using (Tensor x = y) y = functional.avg_pool1d(x, unorderedConvKernelSize, 1, 0, false, false);
					if(remerge is { }){
						using (Tensor x = remerge) remerge = functional.pad(x, (km1, 0), PaddingModes.Zeros, 0.0);
						using (Tensor x = remerge) remerge = functional.avg_pool1d(x, unorderedConvKernelSize, 1, 0, false, false);
						using(remerge){
							using Tensor x = y;
							y = cat(new Tensor[] { remerge, x }, 1);
						}
					}
				}
				using (Tensor x = y) y = x.transpose(0, 1);
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
			finalAttention.L1Regularize(lambda);
			finalCompute.L1Regularize(lambda);
			//Misc.L1RegularizeIMPL(preconv.weight, lambda);

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
			finalAttention.L1Regularize(lambda);
			finalCompute.L1Regularize(lambda);
			//Misc.L1RegularizeIMPL(preconv.weight, lambda);
			Misc.L1RegularizeIMPL(staticPositionalEncoding, lambda);
			Misc.L1RegularizeIMPL(staticInputPositionalEncoding, lambda);
		}
	}


}