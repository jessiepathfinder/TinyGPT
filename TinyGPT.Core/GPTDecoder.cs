
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

			for(int i = 0; i < grulayers; ++i){
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

			if(positionalEncodingStd > 0.0){
				wordEmbeddingSize[0] = max_context_size;
				staticPositionalEncoding = Parameter(normal(0.0, positionalEncodingStd, wordEmbeddingSize));
			} else{
				staticPositionalEncoding = Parameter(zeros(max_context_size,latentTokenSize));
			}
			//staticPositionalEncoding = Parameter(randn(max_context_size, latentTokenSize));
			//long ltk = supplementalWordEmbedding.size(0);

			wordEmbeddingSize[0] = tokenClasses;





			//layers.Add(new TinyRNN("", multipliedWidth, multipliedWidth, epsilon));
			finalAttention = new MultiheadSelfAttention("", latentTokenSize, attentionSize, attentionHeadsCount, epsilon, initialAttentionGain, keyQueryInitGain, auxAttentionDropout);
			//finalBias = Parameter(zeros(1, pretrainedWord2Vec.size(0)));
			wordEmbedding = Parameter(normal(0.0, wordEmbeddingStd, wordEmbeddingSize));
			headcount = attentionHeadsCount;
			//supplementalEngine = Misc.CreateKaimingInitializedLinear((int)supplementalWordEmbedding.size(1), latentTokenSize, true, init.FanInOut.FanIn);
			RegisterComponents();
			this.epsilon = epsilon;
			this.max_context_size = max_context_size;
		}
		public void FinalForward(ref Tensor x){
			using (Tensor y = x) x = finalAttention.forward(y);
			using (Tensor y = x) x = finalCompute.forward(y);
			using (Tensor y = x, x1 = wordEmbedding.transpose(0, 1))
				x = y.matmul(x1);
		}
		public void DecoupledWeightDecayWordEmbeddings(Scalar scalar){
			using(no_grad()){
				wordEmbedding.mul_(scalar);
			}
		}
		public Tensor Forward(ReadOnlySpan<ushort> input, int slice, double dropout = 0.0, bool retearly = false)
		{
			int len = input.Length;
			int maxlen2 = max_context_size;
			if(len > maxlen2 | len == 0)
			{
				throw new ArgumentOutOfRangeException(nameof(input));
			}
			

			long[] longs = new long[len];
			for(int i = 0; i < len; ++i){
				longs[i] = input[i];
			}
			Tensor wordEmbedding = this.wordEmbedding;
			Device device = wordEmbedding.device;
			Scalar epsilon = this.epsilon;


			using (NewDisposeScope())
			{


				Tensor y;
				using (Tensor z2 = tensor(longs, ScalarType.Int64, device, false)){
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
				
				using (Tensor x = y) if(slice > 0)
				{
					using Tensor x2 = y, mask = Transformer.CreateCausalAttentionMask(len - slice, len, ScalarType.Float32, wordEmbedding.device);
					y = finalattention.Forward(x2, slice, mask, dropout);
				} else{
					y = finalattention.Forward(x, 0, null, dropout, true);
				}
				if(retearly){
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


}